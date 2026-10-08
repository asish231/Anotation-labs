const video = document.getElementById('webcam-video');
    const canvas = document.getElementById('stage-canvas');
    const ctx = canvas.getContext('2d');

    let stream = null;
    let isRunning = false;
    let isProcessing = false;
    let lastResult = null;
    let compareMediaPipe = false;

    const FINGER_CHAINS = {
      thumb: [1, 2, 3, 4],
      index: [5, 6, 7, 8],
      middle: [9, 10, 11, 12],
      ring: [13, 14, 15, 16],
      pinky: [17, 18, 19, 20]
    };

    const CHAIN_COLORS = {
      thumb: '#34d399',
      index: '#c084fc',
      middle: '#facc15',
      ring: '#f472b6',
      pinky: '#38bdf8'
    };

    async function initCamera() {
      const hudText = document.getElementById('hud-text');
      const btnCamText = document.getElementById('btn-cam-text');
      const btnToggle = document.getElementById('btn-toggle-cam');

      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        hudText.innerText = 'Camera API not supported or blocked by browser.';
        alert("Webcam access requires a secure context (https:// or http://localhost:8088).\n\nIf you opened http://0.0.0.0:8088, please open http://localhost:8088 instead!");
        btnCamText.innerText = 'Start Camera';
        btnToggle.classList.remove('btn-primary');
        btnToggle.classList.add('btn-secondary');
        return;
      }

      try {
        hudText.innerText = 'Requesting camera access...';

        // 1. Try with user-facing camera first, fallback if constraints fail (common on external webcams / macOS)
        try {
          stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
            audio: false
          });
        } catch (e1) {
          console.warn("Retrying without facingMode constraint...", e1);
          try {
            stream = await navigator.mediaDevices.getUserMedia({
              video: { width: { ideal: 640 }, height: { ideal: 480 } },
              audio: false
            });
          } catch (e2) {
            console.warn("Retrying with fallback video: true...", e2);
            stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
          }
        }

        video.srcObject = stream;

        // 2. Play immediately — do NOT block on onloadedmetadata before calling play(),
        // as WebKit / Safari and modern Chrome won't fire loadedmetadata until play() is invoked!
        try {
          await video.play();
        } catch (playErr) {
          console.warn("video.play() deferred or waiting for user gesture:", playErr);
        }

        // Wait up to 1.5s for video dimensions to become available
        let waitAttempts = 0;
        while ((!video.videoWidth || video.videoWidth === 0) && waitAttempts < 30) {
          await new Promise(r => setTimeout(r, 50));
          waitAttempts++;
        }

        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;
        isRunning = true;

        btnCamText.innerText = 'Stop Camera';
        btnToggle.classList.add('btn-primary');
        btnToggle.classList.remove('btn-secondary');

        hudText.innerText = 'Searching for hand... (Hold hand up to camera)';
        requestAnimationFrame(renderLoop);
      } catch (err) {
        console.error("Camera error:", err);
        isRunning = false;
        btnCamText.innerText = 'Start Camera';
        btnToggle.classList.remove('btn-primary');
        btnToggle.classList.add('btn-secondary');
        hudText.innerText = 'Camera access error: ' + err.message;
        alert(`Camera error: ${err.message}\n\nTroubleshooting:\n1. Make sure you accessed http://localhost:8088 (NOT 0.0.0.0).\n2. Click the camera/tune icon in Chrome's address bar and set Camera to 'Allow'.\n3. Check macOS System Settings > Privacy & Security > Camera to ensure Chrome has camera access.\n4. Make sure no other application (Zoom, FaceTime, Photo Booth) is locking the webcam.`);
      }
    }

    function toggleCamera() {
      if (isRunning) {
        if (stream) {
          stream.getTracks().forEach(t => t.stop());
          stream = null;
        }
        isRunning = false;
        document.getElementById('btn-cam-text').innerText = 'Start Camera';
        document.getElementById('btn-toggle-cam').classList.remove('btn-primary');
        document.getElementById('btn-toggle-cam').classList.add('btn-secondary');
        document.getElementById('hud-text').innerText = 'Camera stopped';
      } else {
        initCamera();
      }
    }

    function toggleCompareMP(val) {
      compareMediaPipe = val;
    }

    // Capture offscreen frame and dispatch to backend
    const captureCanvas = document.createElement('canvas');
    const captureCtx = captureCanvas.getContext('2d');

    async function sendFrameForInference() {
      if (isProcessing || !isRunning || video.readyState < 2) return;
      isProcessing = true;

      captureCanvas.width = video.videoWidth;
      captureCanvas.height = video.videoHeight;
      // Mirror frame for natural self-view
      captureCtx.save();
      captureCtx.scale(-1, 1);
      captureCtx.drawImage(video, -captureCanvas.width, 0, captureCanvas.width, captureCanvas.height);
      captureCtx.restore();

      const b64 = captureCanvas.toDataURL('image/jpeg', 0.82);

      try {
        const resp = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            image: b64,
            compare_mediapipe: compareMediaPipe
          })
        });
        if (resp.ok) {
          const data = await resp.json();
          lastResult = data;
          updateTelemetry(data);
        }
      } catch (err) {
        console.warn("Inference error:", err);
      } finally {
        isProcessing = false;
      }
    }

    function updateTelemetry(data) {
      if (!data) return;

      const badge = document.getElementById('badge-presence');
      const hudText = document.getElementById('hud-text');

      if (data.detected) {
        badge.innerText = 'HAND TRACKED';
        badge.className = 'badge-pass';
        hudText.innerText = `Hand Tracked • Latency: ${data.latency_ms} ms (~${Math.round(1000/data.latency_ms)} FPS)`;
      } else {
        badge.innerText = 'NO HAND';
        badge.className = 'badge-wait';
        hudText.innerText = 'Searching for hand... (Hold hand up to camera)';
      }

      document.getElementById('val-latency').innerText = `${data.latency_ms} ms (~${Math.round(1000/data.latency_ms)} FPS)`;

      if (data.detected && data.biokin) {
        const zRoot = data.biokin.keypoints_3d_mm[0][2];
        document.getElementById('val-depth').innerText = `${Math.round(zRoot)} mm`;

        const angles = data.biokin.joint_angles_deg;
        if (angles && angles.length >= 20) {
          document.getElementById('ang-thumb').innerText = `${Math.round(angles[0][0])}° / ${Math.round(angles[1][0])}°`;
          document.getElementById('ang-index').innerText = `${Math.round(angles[4][0])}° / ${Math.round(angles[5][0])}°`;
          document.getElementById('ang-middle').innerText = `${Math.round(angles[8][0])}° / ${Math.round(angles[9][0])}°`;
          document.getElementById('ang-ring').innerText = `${Math.round(angles[12][0])}° / ${Math.round(angles[13][0])}°`;
          document.getElementById('ang-pinky').innerText = `${Math.round(angles[16][0])}° / ${Math.round(angles[17][0])}°`;
        }
      } else {
        document.getElementById('val-depth').innerText = '-- mm';
        document.getElementById('ang-thumb').innerText = '--° / --°';
        document.getElementById('ang-index').innerText = '--° / --°';
        document.getElementById('ang-middle').innerText = '--° / --°';
        document.getElementById('ang-ring').innerText = '--° / --°';
        document.getElementById('ang-pinky').innerText = '--° / --°';
      }
    }

    function renderLoop() {
      if (!isRunning) return;

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      if (lastResult && lastResult.detected) {
        // Draw ROI Bounding Box corner brackets
        if (document.getElementById('chk-bbox').checked && lastResult.bbox) {
          drawBBoxCorners(lastResult.bbox);
        }

        if (document.getElementById('chk-hud').checked) {
          // Draw optional MediaPipe Ghost (Red dashed)
          if (compareMediaPipe && lastResult.mediapipe) {
            drawSkeleton(lastResult.mediapipe, '#ef4444', 1.5, true);
          }

          // Draw BioKin Guaranteed Skeleton
          if (lastResult.biokin && lastResult.biokin.keypoints_2d) {
            drawBioKinSkeleton(lastResult.biokin.keypoints_2d);
          }
        }
      }

      sendFrameForInference();
      requestAnimationFrame(renderLoop);
    }

    function drawBBoxCorners(bbox) {
      const [x1, y1, x2, y2] = bbox;
      const len = 16;
      ctx.strokeStyle = '#10b981';
      ctx.lineWidth = 2.5;

      // Top-Left
      ctx.beginPath();
      ctx.moveTo(x1, y1 + len); ctx.lineTo(x1, y1); ctx.lineTo(x1 + len, y1);
      ctx.stroke();

      // Top-Right
      ctx.beginPath();
      ctx.moveTo(x2 - len, y1); ctx.lineTo(x2, y1); ctx.lineTo(x2, y1 + len);
      ctx.stroke();

      // Bottom-Left
      ctx.beginPath();
      ctx.moveTo(x1, y2 - len); ctx.lineTo(x1, y2); ctx.lineTo(x1 + len, y2);
      ctx.stroke();

      // Bottom-Right
      ctx.beginPath();
      ctx.moveTo(x2 - len, y2); ctx.lineTo(x2, y2); ctx.lineTo(x2, y2 - len);
      ctx.stroke();

      // Label badge
      ctx.fillStyle = 'rgba(16, 185, 129, 0.2)';
      ctx.fillRect(x1, y1 - 20, 88, 18);
      ctx.fillStyle = '#10b981';
      ctx.font = '600 10px "JetBrains Mono", monospace';
      ctx.fillText('BioKin ROI', x1 + 6, y1 - 7);
    }

    function drawBioKinSkeleton(kps) {
      const p0 = kps[0];

      // 1. Draw Bones
      for (const [finger, chain] of Object.entries(FINGER_CHAINS)) {
        const color = CHAIN_COLORS[finger];
        ctx.strokeStyle = color;
        ctx.lineWidth = 3.5;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        // Root to chain start
        const pBase = kps[chain[0]];
        ctx.beginPath();
        ctx.moveTo(p0[0], p0[1]);
        ctx.lineTo(pBase[0], pBase[1]);
        ctx.stroke();

        // Across finger chain
        ctx.beginPath();
        ctx.moveTo(pBase[0], pBase[1]);
        for (let i = 1; i < chain.length; i++) {
          const pt = kps[chain[i]];
          ctx.lineTo(pt[0], pt[1]);
        }
        ctx.stroke();
      }

      // 2. Draw Joint Spheres
      kps.forEach((pt, idx) => {
        ctx.beginPath();
        const r = (idx === 0) ? 7.5 : 4.5;
        ctx.arc(pt[0], pt[1], r, 0, Math.PI * 2);
        ctx.fillStyle = idx === 0 ? '#FFFFFF' : '#0f172a';
        ctx.fill();
        ctx.lineWidth = 2.0;
        ctx.strokeStyle = idx === 0 ? '#10b981' : '#FFFFFF';
        ctx.stroke();
      });
    }

    function drawSkeleton(kps, color, lineWidth, isDashed = false) {
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      if (isDashed) ctx.setLineDash([4, 4]);

      const p0 = kps[0];
      for (const [finger, chain] of Object.entries(FINGER_CHAINS)) {
        const pBase = kps[chain[0]];
        ctx.beginPath();
        ctx.moveTo(p0[0], p0[1]);
        ctx.lineTo(pBase[0], pBase[1]);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(pBase[0], pBase[1]);
        for (let i = 1; i < chain.length; i++) {
          ctx.lineTo(kps[chain[i]][0], kps[chain[i]][1]);
        }
        ctx.stroke();
      }
      ctx.restore();
    }

    async function exportSnapshot() {
      if (!lastResult || !lastResult.detected || !isRunning) {
        alert("Please ensure your hand is visible and tracked before capturing.");
        return;
      }

      // Capture frame
      captureCanvas.width = video.videoWidth;
      captureCanvas.height = video.videoHeight;
      captureCtx.save();
      captureCtx.scale(-1, 1);
      captureCtx.drawImage(video, -captureCanvas.width, 0, captureCanvas.width, captureCanvas.height);
      captureCtx.restore();

      const b64 = captureCanvas.toDataURL('image/jpeg', 0.95);
      const alertBox = document.getElementById('export-alert');
      alertBox.style.display = 'block';
      alertBox.innerText = 'Saving Label Studio annotation task...';

      try {
        const resp = await fetch('/api/export_snapshot', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            image: b64,
            biokin: lastResult.biokin
          })
        });
        const res = await resp.json();
        if (res.success) {
          alertBox.innerText = `Saved! Task ready: ${res.json_saved}`;
          setTimeout(() => alertBox.style.display = 'none', 5000);
        } else {
          alertBox.innerText = `Error: ${res.error}`;
        }
      } catch (err) {
        alertBox.innerText = `Network error: ${err.message}`;
      }
    }

    async function loadModelList() {
      try {
        const resp = await fetch('/api/list_models');
        if (!resp.ok) return;
        const data = await resp.json();
        const select = document.getElementById('model-select');
        select.innerHTML = '';

        data.models.forEach(m => {
          const opt = document.createElement('option');
          opt.value = m.name;
          opt.innerText = `${m.name} (${m.size_mb} MB • ${m.mtime})`;
          if (m.name === data.current_model) {
            opt.selected = true;
          }
          select.appendChild(opt);
        });

        if (data.current_info) {
          updateModelBadge(data.current_info);
        }
      } catch (err) {
        console.warn("Could not fetch models:", err);
      }
    }

    function updateModelBadge(info) {
      const badge = document.getElementById('val-model-name');
      if (badge && info) {
        badge.innerText = `${info.name} (Ep ${info.epoch})`;
      }
    }

    async function switchModel(modelName) {
      if (!modelName) return;
      const alertBox = document.getElementById('export-alert');
      alertBox.style.display = 'block';
      alertBox.innerText = `Loading checkpoint: ${modelName}...`;

      try {
        const resp = await fetch('/api/reload_model', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ model_name: modelName })
        });
        const res = await resp.json();
        alertBox.innerText = res.message;
        if (res.current_info) {
          updateModelBadge(res.current_info);
        }
        setTimeout(() => alertBox.style.display = 'none', 4000);
      } catch (err) {
        alertBox.innerText = 'Switch error: ' + err.message;
      }
    }

    async function refreshModelList() {
      await loadModelList();
      const alertBox = document.getElementById('export-alert');
      alertBox.style.display = 'block';
      alertBox.innerText = 'Checkpoint list refreshed!';
      setTimeout(() => alertBox.style.display = 'none', 2000);
    }

    async function reloadModelCheckpoint() {
      const btnText = document.getElementById('btn-reload-text');
      btnText.innerText = 'Reloading...';
      const alertBox = document.getElementById('export-alert');
      alertBox.style.display = 'block';
      const currentModel = document.getElementById('model-select').value;
      alertBox.innerText = `Reloading ${currentModel || 'checkpoint'}...`;
      try {
        const resp = await fetch('/api/reload_model', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ model_name: currentModel })
        });
        const res = await resp.json();
        alertBox.innerText = res.message;
        if (res.current_info) {
          updateModelBadge(res.current_info);
        }
        setTimeout(() => alertBox.style.display = 'none', 5000);
      } catch (err) {
        alertBox.innerText = 'Reload error: ' + err.message;
      } finally {
        btnText.innerText = 'Reload Model';
      }
    }

    // Ensure model list and active model badge are populated immediately
    loadModelList();
    document.addEventListener('DOMContentLoaded', loadModelList);

    window.addEventListener('load', () => {
      initCamera();
      loadModelList();
    });