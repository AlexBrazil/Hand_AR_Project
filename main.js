import {
  FilesetResolver,
  HandLandmarker
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.12";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

const PALM_INDICES = [5, 9, 13, 17];
const depthScale = 420;
const heartOffset = 40;
let scaleMultiplier = 4;

const scaleSlider = document.getElementById("scaleSlider");
const scaleValue = document.getElementById("scaleValue");

const video = document.getElementById("camera");
const overlayCanvas = document.getElementById("overlay");
const statusElement = document.getElementById("status");
const startButton = document.getElementById("startButton");

let handLandmarker = null;
let animationFrame = null;
let lastVideoTime = -1;
let modelLoaded = false;
let running = false;

const threeScene = new THREE.Scene();
const renderer = new THREE.WebGLRenderer({ canvas: overlayCanvas, alpha: true, antialias: true });
renderer.setPixelRatio(window.devicePixelRatio ?? 1);
renderer.setClearColor(0x000000, 0);

const ambient = new THREE.AmbientLight(0xffffff, 0.6);
threeScene.add(ambient);

const directional = new THREE.DirectionalLight(0xffffff, 0.9);
directional.position.set(0, 0, 1);
threeScene.add(directional);

let orthoCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, -1000, 1000);
orthoCamera.position.z = 500;

let renderWidth = 0;
let renderHeight = 0;

const heartGroup = new THREE.Group();
heartGroup.visible = false;
threeScene.add(heartGroup);

const loader = new GLTFLoader();
// const heartUrl = new URL("./cenas/heart/scene.gltf", import.meta.url);
const heartUrl = new URL("/cenas/heart/scene.gltf", import.meta.url);
loader.load(
  heartUrl.href,
  (gltf) => {
    const heart = gltf.scene;
    const bbox = new THREE.Box3().setFromObject(heart);
    const center = new THREE.Vector3();
    bbox.getCenter(center);
    heart.position.sub(center);

    const size = new THREE.Vector3();
    bbox.getSize(size);
    const maxAxis = Math.max(size.x, size.y, size.z) || 1;
    heart.scale.setScalar(80 / maxAxis);

    heartGroup.add(heart);
    modelLoaded = true;
  },
  undefined,
  (error) => console.error("Falha ao carregar modelo GLTF", error)
);

const averagePoint = (indices, landmarks) => {
  const sum = indices.reduce(
    (acc, index) => {
      const p = landmarks[index];
      acc.x += p.x;
      acc.y += p.y;
      acc.z += p.z ?? 0;
      return acc;
    },
    { x: 0, y: 0, z: 0 }
  );

  const count = indices.length || 1;
  return {
    x: sum.x / count,
    y: sum.y / count,
    z: sum.z / count
  };
};

const landmarkToScene = (landmark) => {
  return new THREE.Vector3(
    landmark.x * renderWidth - renderWidth / 2,
    renderHeight / 2 - landmark.y * renderHeight,
    (landmark.z ?? 0) * depthScale
  );
};

const computeBasis = (landmarks) => {
  const wrist = landmarkToScene(landmarks[0]);
  const indexBase = landmarkToScene(landmarks[5]);
  const pinkyBase = landmarkToScene(landmarks[17]);
  const middleBase = landmarkToScene(landmarks[9]);

  const indexVec = indexBase.clone().sub(wrist);
  const pinkyVec = pinkyBase.clone().sub(wrist);
  const normal = new THREE.Vector3().crossVectors(indexVec, pinkyVec);

  if (normal.lengthSq() === 0) {
    normal.set(0, 0, -1);
  } else if (normal.z > 0) {
    normal.multiplyScalar(-1);
  }
  normal.normalize();

  const up = middleBase.clone().sub(wrist).normalize();
  const right = new THREE.Vector3().crossVectors(up, normal).normalize();
  up.crossVectors(normal, right).normalize();

  return { normal, up };
};

const ensureRendererSize = () => {
  const needsResize =
    renderWidth !== video.videoWidth || renderHeight !== video.videoHeight;

  if (!needsResize && renderWidth && renderHeight) {
    return;
  }

  renderWidth = video.videoWidth;
  renderHeight = video.videoHeight;

  overlayCanvas.width = renderWidth;
  overlayCanvas.height = renderHeight;
  renderer.setSize(renderWidth, renderHeight, false);

  orthoCamera.left = -renderWidth / 2;
  orthoCamera.right = renderWidth / 2;
  orthoCamera.top = renderHeight / 2;
  orthoCamera.bottom = -renderHeight / 2;
  orthoCamera.updateProjectionMatrix();
};

const setStatus = (message) => {
  statusElement.textContent = message;
};

const updateHeartPose = (landmarks) => {
  if (!modelLoaded) {
    return;
  }

  const center = averagePoint(PALM_INDICES, landmarks);
  const centerScene = landmarkToScene(center);
  const { normal, up } = computeBasis(landmarks);

  const palmWidthPixels = Math.hypot(
    (landmarks[5].x - landmarks[17].x) * renderWidth,
    (landmarks[5].y - landmarks[17].y) * renderHeight
  );
  const baseScale = THREE.MathUtils.clamp(palmWidthPixels / 180, 0.3, 2.5);
  const scale = THREE.MathUtils.clamp(baseScale * scaleMultiplier, 0.2, 8);
  heartGroup.scale.setScalar(scale);

  heartGroup.position.copy(
    centerScene.clone().add(normal.clone().multiplyScalar(heartOffset))
  );
  heartGroup.up.copy(up);
  heartGroup.lookAt(heartGroup.position.clone().sub(normal));
  heartGroup.visible = true;
};

const predictLoop = () => {
  if (!running) {
    return;
  }

  ensureRendererSize();

  if (video.currentTime === lastVideoTime) {
    animationFrame = requestAnimationFrame(predictLoop);
    return;
  }

  lastVideoTime = video.currentTime;

  const now = performance.now();
  const results = handLandmarker.detectForVideo(video, now);
  const hands = results.landmarks ?? [];

  if (hands.length > 0) {
    updateHeartPose(hands[0]);
  } else {
    heartGroup.visible = false;
  }

  renderer.render(threeScene, orthoCamera);
  animationFrame = requestAnimationFrame(predictLoop);
};

const stop = () => {
  running = false;
  cancelAnimationFrame(animationFrame);
  animationFrame = null;
  heartGroup.visible = false;
  renderer.render(threeScene, orthoCamera);
  if (video.srcObject) {
    video.srcObject.getTracks().forEach((track) => track.stop());
    video.srcObject = null;
  }
  startButton.disabled = false;
  startButton.textContent = "Ativar Webcam";
  setStatus("Webcam desligada");
};

const start = async () => {
  if (!handLandmarker) {
    setStatus("Carregando modelo...");
    return;
  }

  try {
    startButton.disabled = true;
    setStatus("Solicitando acesso à webcam...");
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
      audio: false
    });
    video.srcObject = stream;

    await new Promise((resolve) => {
      if (video.readyState >= 3) {
        resolve();
      } else {
        video.onloadeddata = () => resolve();
      }
    });

    await video.play();

    running = true;
    startButton.textContent = "Desativar Webcam";
    setStatus("Posicione a palma da mão diante da câmera");
    ensureRendererSize();
    predictLoop();
  } catch (error) {
    console.error(error);
    setStatus("Não foi possível acessar a webcam");
    startButton.disabled = false;
  }
};

startButton.addEventListener("click", () => {
  if (running) {
    stop();
  } else {
    start();
  }
});

if (scaleSlider && scaleValue) {
  scaleSlider.addEventListener("input", (event) => {
    scaleMultiplier = Number(event.target.value);
    scaleValue.textContent = `${scaleMultiplier.toFixed(2)}x`;
  });
}

const bootstrap = async () => {
  if (!navigator.mediaDevices?.getUserMedia) {
    startButton.disabled = true;
    setStatus("getUserMedia não suportado neste navegador");
    return;
  }

  setStatus("Carregando Hand Landmarker...");

  const fileset = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.12/wasm"
  );

  handLandmarker = await HandLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    },
    runningMode: "VIDEO",
    numHands: 1
  });

  setStatus("Clique em \"Ativar Webcam\" para iniciar");
};

bootstrap();

window.addEventListener("beforeunload", stop);
