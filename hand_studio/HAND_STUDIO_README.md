# Hand Studio — our forked Label Studio for hand annotation

A fork of Label Studio v1.23.2 (tag `1.23.2`, commit `df96d79`) dedicated to
21-point hand-skeleton annotation. Runs on the existing `mp_env` virtualenv.

## What's inside

- `label_studio/` — forked backend source (owned by us, fully customizable)
- `label_studio/core/static_build/` — prebuilt static assets (copied from the pip wheel)
- `web/dist/` — prebuilt frontend bundles: main React app, editor, data manager
- `hand_studio_data/` — isolated data dir (SQLite DB, media uploads)
- `../scripts/setup_hand_studio.py` — idempotent bootstrap script

## Launch

```bash
cd hand_studio
BASE_DATA_DIR="$PWD/hand_studio_data" \
../mp_env/bin/python label_studio/manage.py runserver 0.0.0.0:8080 --noreload
```

Then open http://localhost:8080

## Accounts

| Role      | Email                        | Password         |
|-----------|------------------------------|------------------|
| Admin     | admin@handstudio.local       | HandStudio#2026  |
| Annotator | annotator@handstudio.local   | Annotator#2026   |

## Annotator workflow

1. **Login** → dashboard shows the "Hand Keypoint Annotation" project
2. Open project → **Data Manager** (20 FreiHAND tasks) or **Label All Tasks**
3. Each task opens with the **MediaPipe pre-annotation** (21 keypoints) already
   on the image
4. The predictions panel lets you switch to **DWPose** (second model) or
   **Compare All** to cross-check both models while fixing points
5. Fix keypoints to 100% precision → **Submit** → task is stored and the
   queue advances to the next task
6. Completed tasks stay visible in the Data Manager (annotator's history)

## Re-seed / extend tasks

```bash
mp_env/bin/python scripts/setup_hand_studio.py --num-images 50
```

Idempotent: adds new tasks up to `--num-images`, backfills FileUpload records,
re-runs MediaPipe + DWPose predictions. Task images come from
`biokin_data/freihand/evaluation/rgb/`.

## Notes / limitations

- Frontend is the prebuilt 1.23.2 bundle from the pip wheel; rebuilding the UI
  from `web/` source requires [bun](https://bun.sh) (`cd web && bun install && bun run build`)
- Branding still says "Label Studio" — rebranding is a follow-up
- The annotation UI is Label Studio's proven editor; custom "My Work" pages,
  custom branding and pipeline improvements layer on top of this fork
