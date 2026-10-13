# PushUpSense Mobile (prototype)

A mobile-first web prototype that uses MediaPipe Pose Landmarker in the browser.

## Files
- `index.html` — page structure
- `style.css` — mobile-first styling
- `app.js` — camera, pose landmarks, angle calculations, rep counter, feedback

## Run locally
Camera access requires HTTPS or localhost.

Option A (Python):
1. Open a terminal in this folder.
2. Run `python -m http.server 8000`
3. Open `http://localhost:8000` in a desktop browser.

Option B (VS Code):
Use the Live Server extension and open the page at the localhost URL it provides.

For testing on a phone, `localhost` on your PC does not automatically work from the phone. Deploy the folder to an HTTPS static host, or use a secure local tunnel.

## Important limitations
- This is an initial prototype; test on actual Android and iPhone browsers before using it for a survey.
- It loads the MediaPipe Tasks Vision library and model from online URLs, so internet is needed on first load.
- Browser pose inference can be slower on some phones. If GPU delegate initialization fails on a device, try removing `delegate: "GPU"` from `app.js` to let MediaPipe use its default.
- The form rules reproduce the broad thresholds in the supplied Python files, but JavaScript/browser landmark coordinates may not exactly match the Python version. Validate with test reps before collecting survey results.
- The camera stream is processed in the browser and is not intentionally recorded or uploaded by this code. The CDN/model are external dependencies; disclose camera usage and obtain participant consent before testing.
- The current version is a prototype, not a medical or professional coaching tool.
