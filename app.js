import { FilesetResolver, PoseLandmarker, DrawingUtils } from
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";
const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const VISIBILITY_THRESHOLD = 0.5;
const UP_ANGLE = 160;
const DOWN_ANGLE = 90;
const MAX_FRONT_ANGLE_DIFFERENCE = 45;
const SIDE_HIP_ANGLE = 160;
const SIDE_SHOULDER_ANGLE = 40;

const video = document.querySelector("#video");
const canvas = document.querySelector("#overlay");
const ctx = canvas.getContext("2d");
const cameraPlaceholder = document.querySelector("#cameraPlaceholder");
const startBtn = document.querySelector("#startBtn");
const resetBtn = document.querySelector("#resetBtn");
const message = document.querySelector("#message");
const cameraStatus = document.querySelector("#cameraStatus");
const poseStatus = document.querySelector("#poseStatus");
const feedbackEl = document.querySelector("#feedback");
const feedbackHint = document.querySelector("#feedbackHint");
const formIcon = document.querySelector("#formIcon");
const repCountEl = document.querySelector("#repCount");
const elbowAngleEl = document.querySelector("#elbowAngle");
const viewChip = document.querySelector("#viewChip");

let landmarker = null;
let stream = null;
let running = false;
let viewMode = "side";
let lastVideoTime = -1;
let reps = 0;
let phase = "waiting_up";
let drawingUtils = null;
let busy = false;

function angle2D(a, b, c) {
  const v1 = [a.x - b.x, a.y - b.y];
  const v2 = [c.x - b.x, c.y - b.y];
  const len1 = Math.hypot(...v1);
  const len2 = Math.hypot(...v2);
  if (!len1 || !len2) return null;
  const cosine = (v1[0] * v2[0] + v1[1] * v2[1]) / (len1 * len2);
  return Math.acos(Math.max(-1, Math.min(1, cosine))) * 180 / Math.PI;
}

function angle3D(a, b, c) {
  const v1 = [a.x - b.x, a.y - b.y, a.z - b.z];
  const v2 = [c.x - b.x, c.y - b.y, c.z - b.z];
  const len1 = Math.hypot(...v1);
  const len2 = Math.hypot(...v2);
  if (!len1 || !len2) return null;
  const cosine = (v1.reduce((s, v, i) => s + v * v2[i], 0)) / (len1 * len2);
  return Math.acos(Math.max(-1, Math.min(1, cosine))) * 180 / Math.PI;
}

function reliableAngle(landmarks, ids, useDepth = false) {
  if (!ids.every(id => (landmarks[id]?.visibility ?? 0) >= VISIBILITY_THRESHOLD)) return null;
  const pts = ids.map(id => landmarks[id]);
  const result = useDepth ? angle3D(...pts) : angle2D(...pts);
  return Number.isFinite(result) ? result : null;
}

function updateCounter(angle) {
  if (!Number.isFinite(angle)) return;
  if (phase === "waiting_up") {
    if (angle >= UP_ANGLE) phase = "up";
  } else if (phase === "up") {
    if (angle <= DOWN_ANGLE) phase = "down";
  } else if (phase === "down" && angle >= UP_ANGLE) {
    reps += 1;
    phase = "up";
    repCountEl.textContent = String(reps);
  }
}

function setFeedback(text, hint = "") {
  feedbackEl.textContent = text;
  feedbackHint.textContent = hint;
  const good = text === "GOOD FORM";
  formIcon.textContent = good ? "✓" : text === "FIX FORM" ? "!" : "↗";
  formIcon.style.background = good ? "#2c3a22" : text === "FIX FORM" ? "#3b2c1b" : "#253020";
  formIcon.style.color = good ? "#c6f36b" : text === "FIX FORM" ? "#ffcb78" : "#c6f36b";
}

function evaluateSide(landmarks) {
  const required = [11, 13, 15, 23, 25];
  if (!required.every(id => (landmarks[id]?.visibility ?? 0) >= VISIBILITY_THRESHOLD)) {
    return { feedback: "NO CLEAR POSE", hint: "Move farther back and keep your shoulder, elbow, wrist, hip, and knee visible.", angle: null };
  }
  const elbow = reliableAngle(landmarks, [11, 13, 15]);
  const shoulder = reliableAngle(landmarks, [13, 11, 23]);
  const hip = reliableAngle(landmarks, [11, 23, 25]);
  if ([elbow, shoulder, hip].some(v => v === null)) {
    return { feedback: "NO CLEAR POSE", hint: "Adjust your position so your full side profile is visible.", angle: null };
  }
  const good = (elbow <= DOWN_ANGLE && hip > SIDE_HIP_ANGLE) ||
    (elbow >= UP_ANGLE && shoulder > SIDE_SHOULDER_ANGLE && hip > SIDE_HIP_ANGLE);
  return { feedback: good ? "GOOD FORM" : "FIX FORM",
    hint: good ? "Keep your body aligned and move through a controlled range." : "Keep your hips aligned and control the elbow bend.",
    angle: elbow, angles: [elbow, shoulder, hip] };
}

function evaluateFront(landmarks) {
  const left = reliableAngle(landmarks, [11, 13, 15], true);
  const right = reliableAngle(landmarks, [12, 14, 16], true);
  const valid = [left, right].filter(v => v !== null);
  if (!valid.length) return { feedback: "NO CLEAR POSE", hint: "Face the camera and keep both arms visible.", angle: null };
  let elbow = valid.length === 2 ? (left + right) / 2 : valid[0];
  let feedback;
  if (valid.length === 2 && Math.abs(left - right) > MAX_FRONT_ANGLE_DIFFERENCE) {
    feedback = "FIX FORM";
  } else if (valid.length === 2 &&
      ((left <= DOWN_ANGLE && right <= DOWN_ANGLE) || (left >= UP_ANGLE && right >= UP_ANGLE))) {
    feedback = "GOOD FORM";
  } else if (valid.length === 1 && (elbow <= DOWN_ANGLE || elbow >= UP_ANGLE)) {
    feedback = "GOOD FORM";
  } else {
    feedback = "FIX FORM";
  }
  return { feedback, hint: feedback === "GOOD FORM" ? "Try to keep both arms moving evenly." : "Check that both arms bend and straighten together.", angle: elbow, angles: [left, right] };
}

function drawPose(landmarks) {
  const width = video.videoWidth;
  const height = video.videoHeight;
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  // The video is mirrored in CSS, so mirror the canvas drawing to line up with it.
  ctx.save();
  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);
  drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, {
    color: "#c6f36b", lineWidth: 3
  });
  drawingUtils.drawLandmarks(landmarks, {
    color: "#ffffff", fillColor: "#c6f36b", lineWidth: 1, radius: 3
  });
  ctx.restore();
}

function renderLoop() {
  if (!running || busy) {
    if (running) requestAnimationFrame(renderLoop);
    return;
  }
  if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    busy = true;
    try {
      const result = landmarker.detectForVideo(video, performance.now());
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const landmarks = result.landmarks?.[0];
      if (landmarks) {
        drawPose(landmarks);
        poseStatus.textContent = "POSE FOUND";
        const check = viewMode === "side" ? evaluateSide(landmarks) : evaluateFront(landmarks);
        setFeedback(check.feedback, check.hint);
        elbowAngleEl.textContent = check.angle === null ? "--" : String(Math.round(check.angle));
        if (check.angle !== null) updateCounter(check.angle);
      } else {
        poseStatus.textContent = "NO POSE";
        setFeedback("NO POSE DETECTED", "Step back and make sure your body is visible.");
        elbowAngleEl.textContent = "--";
      }
      lastVideoTime = video.currentTime;
    } catch (error) {
      console.error(error);
      message.textContent = "Pose detection hit an error. Try refreshing the page.";
    } finally {
      busy = false;
    }
  }
  if (running) requestAnimationFrame(renderLoop);
}

async function startCamera() {
  if (running) {
    stopCamera();
    return;
  }
  startBtn.disabled = true;
  startBtn.textContent = "Starting camera…";
  message.textContent = "Loading pose model and requesting camera access…";
  try {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("This browser requires a secure HTTPS page (or localhost) for camera access.");
    }
    if (!landmarker) {
      const vision = await FilesetResolver.forVisionTasks(WASM_URL);
      landmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
        runningMode: "VIDEO",
        numPoses: 1,
        minPoseDetectionConfidence: 0.5,
        minPosePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5
      });
      drawingUtils = new DrawingUtils(ctx);
    }
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } }
    });
    video.srcObject = stream;
    await video.play();
    cameraPlaceholder.hidden = true;
    running = true;
    cameraStatus.textContent = "Camera running";
    startBtn.textContent = "Stop camera";
    startBtn.disabled = false;
    message.textContent = "Camera is active. Position your full upper body in the frame.";
    lastVideoTime = -1;
    requestAnimationFrame(renderLoop);
  } catch (error) {
    console.error(error);
    message.textContent = error?.message || "Could not start the camera. Check browser permissions and try again.";
    startBtn.textContent = "Start camera";
    startBtn.disabled = false;
    if (stream) stream.getTracks().forEach(track => track.stop());
    stream = null;
  }
}

function stopCamera() {
  running = false;
  if (stream) stream.getTracks().forEach(track => track.stop());
  stream = null;
  video.srcObject = null;
  cameraPlaceholder.hidden = false;
  cameraStatus.textContent = "Camera not started";
  poseStatus.textContent = "WAITING";
  startBtn.textContent = "Start camera";
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  setFeedback("Ready when you are", "Keep your full body visible in frame.");
  message.textContent = "Camera stopped. Your video is not saved by this prototype.";
}

startBtn.addEventListener("click", startCamera);
resetBtn.addEventListener("click", () => {
  reps = 0;
  phase = "waiting_up";
  repCountEl.textContent = "0";
  message.textContent = "Count reset. Start in the top push-up position for best results.";
});
document.querySelectorAll(".mode-btn").forEach(button => {
  button.addEventListener("click", () => {
    viewMode = button.dataset.mode;
    document.querySelectorAll(".mode-btn").forEach(b => b.classList.toggle("active", b === button));
    viewChip.textContent = `${viewMode.toUpperCase()} VIEW`;
    message.textContent = viewMode === "side"
      ? "Side view: keep shoulder, elbow, wrist, hip, and knee visible."
      : "Front view: face the camera and keep both elbows and wrists visible.";
    setFeedback("Ready for " + viewMode + " view", "Start from the top position when counting reps.");
    phase = "waiting_up";
  });
});
window.addEventListener("pagehide", stopCamera);
