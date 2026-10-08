"""
Hand Studio Setup — bootstraps our forked Label Studio (hand_studio/) into a
hand-annotation-specific studio:

1. Ensures admin + annotator accounts exist
2. Creates the "Hand Keypoint Annotation" project with a 21-point labeling config
3. Imports a batch of FreiHAND images as tasks
4. Runs MediaPipe (default) + DWPose (toggle layer) and stores both predictions
   per task, so annotators see MediaPipe pre-annotation by default and can
   switch to the DWPose overlay in the editor.

Run:  python scripts/setup_hand_studio.py [--num-images 20]
"""
import os
import sys
import shutil
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LS_SRC = os.path.join(PROJECT_ROOT, "hand_studio", "label_studio")
DATA_DIR = os.path.join(PROJECT_ROOT, "hand_studio", "hand_studio_data")

# Must be set BEFORE Django setup
sys.path.insert(0, LS_SRC)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings.label_studio")
os.environ.setdefault("BASE_DATA_DIR", DATA_DIR)

import django

django.setup()

from django.conf import settings
from users.models import User
from organizations.models import Organization, OrganizationMember
from projects.models import Project
from tasks.models import Task, Prediction

ADMIN_EMAIL = "admin@handstudio.local"
ADMIN_PASSWORD = "HandStudio#2026"
ANNOTATOR_EMAIL = "annotator@handstudio.local"
ANNOTATOR_PASSWORD = "Annotator#2026"

LANDMARK_NAMES = [
    "0_wrist",
    "1_thumb_cmc", "2_thumb_mcp", "3_thumb_ip", "4_thumb_tip",
    "5_index_mcp", "6_index_pip", "7_index_dip", "8_index_tip",
    "9_middle_mcp", "10_middle_pip", "11_middle_dip", "12_middle_tip",
    "13_ring_mcp", "14_ring_pip", "15_ring_dip", "16_ring_tip",
    "17_pinky_mcp", "18_pinky_pip", "19_pinky_dip", "20_pinky_tip",
]

# Finger colors matching the reference hand-skeleton style:
# thumb=green, index=purple, middle=cyan, ring=pink, pinky=blue, wrist=white
FINGER_COLORS = {
    "thumb": "#00E676",   # green
    "index": "#B388FF",   # purple
    "middle": "#40C4FF",  # cyan
    "ring": "#F06292",    # pink
    "pinky": "#448AFF",   # blue
}

def _color_for_joint(idx, name):
    if idx == 0:
        return "#FFFFFF"  # wrist
    parts = name.split("_")
    finger = parts[1] if len(parts) > 2 else None  # "1_thumb_cmc" -> "thumb"
    return FINGER_COLORS.get(finger, "#CCCCCC")

LABEL_COLORS = [_color_for_joint(i, n) for i, n in enumerate(LANDMARK_NAMES)]


def get_labeling_config():
    labels = "\n".join(
        f'    <Label value="{name}" background="{LABEL_COLORS[i % len(LABEL_COLORS)]}"/>'
        for i, name in enumerate(LANDMARK_NAMES)
    )
    return f"""<View>
  <Image name="img" value="$image" zoom="true" zoomControl="true" maxWidth="100%" maxHeight="100%"/>
  <KeyPointLabels name="kp" toName="img" strokeWidth="4">
{labels}
  </KeyPointLabels>
</View>"""


def ensure_user(email, password, is_admin=False):
    user = User.objects.filter(email=email).first()
    if user is None:
        user = User.objects.create(email=email, username=email)
        user.set_password(password)
        user.is_staff = is_admin
        user.is_superuser = is_admin
        user.save()
        print(f"[users] created {'admin' if is_admin else 'annotator'}: {email}")
    else:
        print(f"[users] exists: {email}")
    # Every user MUST belong to an organization, otherwise login 500s
    org = Organization.objects.first()
    if org is None:
        org = Organization.objects.create(title="Hand Studio", created_by=user)
        print("[users] created organization 'Hand Studio'")
    if not OrganizationMember.objects.filter(user=user, organization=org, deleted_at__isnull=True).exists():
        org.add_user(user)
        # Make it the active organization so login succeeds
        user.active_organization = org
        user.save(update_fields=["active_organization"])
        print(f"[users] added {email} to organization")
    return user


def ensure_project(admin):
    project = Project.objects.filter(title="Hand Keypoint Annotation").first()
    config = get_labeling_config()
    if project is None:
        org = Organization.objects.first()
        project = Project.objects.create(
            title="Hand Keypoint Annotation",
            description="21-point hand skeleton annotation. Pre-annotated by MediaPipe; switch predictions to compare DWPose.",
            created_by=admin,
            organization=org,
            label_config=config,
        )
        project.organization = org
        project.save(update_fields=["organization"])
        print("[project] created 'Hand Keypoint Annotation'")
    else:
        project.label_config = config
        project.save()
        print("[project] exists (config refreshed)")
    # The project must belong to the org and be published, or annotators can't see it
    org = Organization.objects.first()
    changed = []
    if project.organization_id != org.id:
        project.organization = org
        changed.append("organization")
    if not getattr(project, "is_published", True):
        project.is_published = True
        changed.append("is_published")
    if changed:
        project.save()
        print(f"[project] fixed: {', '.join(changed)}")
    return project


def import_freihand_tasks(project, num_images):
    from data_import.models import FileUpload

    media_upload = os.path.join(settings.BASE_DATA_DIR, "media", "upload")
    os.makedirs(media_upload, exist_ok=True)
    src_dir = os.path.join(PROJECT_ROOT, "biokin_data", "freihand", "evaluation", "rgb")
    admin = User.objects.get(email=ADMIN_EMAIL)

    existing = Task.objects.filter(project=project).count()
    if existing < num_images:
        names = sorted(os.listdir(src_dir))[:num_images]
        for name in names:
            dst_name = f"freihand_{name}"
            dst = os.path.join(media_upload, dst_name)
            if not os.path.exists(dst):
                shutil.copyfile(os.path.join(src_dir, name), dst)
            task = Task.objects.create(
                project=project,
                data={"image": f"/data/upload/{dst_name}"},
                is_labeled=False,
            )
            fu = FileUpload(user=admin, project=project)
            fu.file.name = f"upload/{dst_name}"
            fu.save()
            task.file_upload = fu
            task.save(update_fields=["file_upload"])
        print(f"[tasks] imported {len(names)} FreiHAND images as tasks")
    else:
        print(f"[tasks] project already has {existing} tasks — skipping import")

    # Backfill FileUpload records for tasks imported before this fix
    missing = 0
    for task in Task.objects.filter(project=project):
        img = task.data.get("image", "")
        if not img.startswith("/data/upload/"):
            continue
        rel = f"upload/{img[len('/data/upload/'):]}"
        if not FileUpload.objects.filter(project=project, file=rel).exists():
            fu = FileUpload(user=admin, project=project)
            fu.file.name = rel
            fu.save()
            if not getattr(task, "file_upload_id", None):
                task.file_upload = fu
                task.save(update_fields=["file_upload"])
            missing += 1
    if missing:
        print(f"[tasks] backfilled {missing} FileUpload records")


def format_keypoints(kps, w, h, prefix):
    results = []
    for idx in range(min(21, len(kps))):
        px, py = kps[idx]
        results.append({
            "original_width": w,
            "original_height": h,
            "image_rotation": 0,
            "value": {
                "x": round(float(px) / w * 100, 3),
                "y": round(float(py) / h * 100, 3),
                "width": 0.5,
                "keypointlabels": [LANDMARK_NAMES[idx]],
            },
            "id": f"{prefix}_{idx}",
            "from_name": "kp",
            "to_name": "img",
            "type": "keypointlabels",
        })
    return results


def read_task_image(task):
    img_rel = task.data.get("image", "")
    media_root = os.path.join(settings.BASE_DATA_DIR, "media")
    if img_rel.startswith("/data/"):
        return os.path.join(media_root, img_rel[len("/data/"):])
    return img_rel


def run_models(project):
    import cv2
    import numpy as np
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
    from rtmlib import Wholebody

    tasks = list(Task.objects.filter(project=project).order_by("id"))
    print(f"[models] running MediaPipe + DWPose on {len(tasks)} tasks...")

    landmarker = HandLandmarker.create_from_options(HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.join(PROJECT_ROOT, "hand_landmarker.task")),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.1,
    ))

    wb = Wholebody(mode="lightweight", device="cpu")

    done = 0
    for task in tasks:
        img_path = read_task_image(task)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"  task {task.id}: could not read {img_path}")
            continue
        h, w = img_bgr.shape[:2]

        # --- MediaPipe (default pre-annotation) ---
        mp_res, mp_score = [], 0.0
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        det = landmarker.detect(mp_img)
        if det.hand_landmarks:
            lms = det.hand_landmarks[0]
            mp_score = 0.95
            for idx, pt in enumerate(lms):
                mp_res.append({
                    "original_width": w, "original_height": h, "image_rotation": 0,
                    "value": {
                        "x": round(pt.x * 100, 3), "y": round(pt.y * 100, 3),
                        "width": 0.5, "keypointlabels": [LANDMARK_NAMES[idx]],
                    },
                    "id": f"mp_{idx}", "from_name": "kp", "to_name": "img",
                    "type": "keypointlabels",
                })

        # --- DWPose (toggle/compare layer) ---
        dw_res, dw_score = [], 0.0
        try:
            kps, scs = wb(img_bgr)
            if kps is not None and len(kps) > 0:
                lh_kps, lh_sc = kps[0, 91:112], float(np.mean(scs[0, 91:112]))
                rh_kps, rh_sc = kps[0, 112:133], float(np.mean(scs[0, 112:133]))
                if rh_sc >= lh_sc and rh_sc > 0.15:
                    dw_score, dw_res = round(rh_sc, 3), format_keypoints(rh_kps, w, h, "dw")
                elif lh_sc > 0.15:
                    dw_score, dw_res = round(lh_sc, 3), format_keypoints(lh_kps, w, h, "dw")
        except Exception as e:
            print(f"  task {task.id}: DWPose error: {e}")

        Prediction.objects.filter(task=task).delete()
        for model_name, res, score in (
            ("mediapipe", mp_res, mp_score),
            ("dwpose", dw_res, dw_score),
        ):
            if res:
                Prediction.objects.create(
                    task=task, project=project, result=res, score=score,
                    model_version=model_name,
                )
        done += 1
        print(f"  task {task.id}: mediapipe={len(mp_res)}pts@{mp_score} dwpose={len(dw_res)}pts@{dw_score}")

    landmarker.close()
    print(f"[models] done for {done}/{len(tasks)} tasks")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-images", type=int, default=20)
    args = parser.parse_args()

    print(f"[setup] data dir: {settings.BASE_DATA_DIR}")
    admin = ensure_user(ADMIN_EMAIL, ADMIN_PASSWORD, is_admin=True)
    ensure_user(ANNOTATOR_EMAIL, ANNOTATOR_PASSWORD, is_admin=False)
    project = ensure_project(admin)
    import_freihand_tasks(project, args.num_images)
    run_models(project)

    n_tasks = Task.objects.filter(project=project).count()
    n_preds = Prediction.objects.filter(project=project).count()
    print(f"\n[setup] DONE — tasks: {n_tasks}, predictions: {n_preds}")
    print(f"[setup] Studio URL: http://localhost:8080")
    print(f"[setup] Admin:     {ADMIN_EMAIL} / {ADMIN_PASSWORD}")
    print(f"[setup] Annotator: {ANNOTATOR_EMAIL} / {ANNOTATOR_PASSWORD}")


if __name__ == "__main__":
    main()
