"""Face detection: YuNet + dlib 68-point landmarks with EMA smoothing and head pose."""
from __future__ import annotations
import logging, os, time as _time
from dataclasses import dataclass
from typing import Optional
import cv2, dlib, numpy as np
from config import AppConfig

logger = logging.getLogger(__name__)

_YUNET = os.path.join(os.path.dirname(cv2.__file__), "data", "face_detection_yunet_2023mar.onnx")
_SP = os.path.join(os.path.dirname(__file__), "assets", "models", "shape_predictor_68_face_landmarks.dat")
LEFT_EYE, RIGHT_EYE = list(range(36, 42)), list(range(42, 48))

# 3D model points for head pose estimation (average human face)
_MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),       # Nose tip (30)
    (0.0, -63.6, -12.5),    # Chin (8)
    (-43.3, 32.7, -26.0),   # Left eye left corner (36)
    (43.3, 32.7, -26.0),    # Right eye right corner (45)
    (-28.9, -28.9, -24.1),  # Left mouth corner (48)
    (28.9, -28.9, -24.1),   # Right mouth corner (54)
], dtype=np.float64)

# 2D landmark indices corresponding to model points
_LANDMARK_IDXS = [30, 8, 36, 45, 48, 54]

def compute_ear(lm, idx):
    pts = lm[idx]
    return float((np.linalg.norm(pts[1]-pts[5]) + np.linalg.norm(pts[2]-pts[4])) / (2*np.linalg.norm(pts[0]-pts[3]) + 1e-6))

def estimate_head_pose(landmarks, frame_shape):
    """Estimate head pose (pitch, yaw, roll) from 68-point landmarks."""
    h, w = frame_shape[:2]
    image_points = np.array([
        landmarks[30],  # Nose tip
        landmarks[8],   # Chin
        landmarks[36],  # Left eye left corner
        landmarks[45],  # Right eye right corner
        landmarks[48],  # Left mouth corner
        landmarks[54],  # Right mouth corner
    ], dtype=np.float64)

    # Camera matrix approximation
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))

    success, rvec, tvec = cv2.solvePnP(
        _MODEL_POINTS, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE)

    if not success:
        return 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rvec)

    # Project nose tip forward to get projection matrix
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

    pitch = angles[0]  # Up/down nod
    yaw = angles[1]    # Left/right turn
    roll = angles[2]   # Tilt

    return float(pitch), float(yaw), float(roll)


@dataclass
class FaceResult:
    detected: bool = False
    landmarks: Optional[np.ndarray] = None
    bbox: Optional[tuple] = None
    confidence: float = 0.0
    left_ear: float = 0.0
    right_ear: float = 0.0
    avg_ear: float = 0.0
    yawning_ratio: float = 0.0
    is_yawning: bool = False
    frame_timestamp: float = 0.0
    head_pitch: float = 0.0
    head_yaw: float = 0.0
    head_roll: float = 0.0

class FaceDetector:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        if not os.path.exists(_YUNET):
            import urllib.request
            os.makedirs(os.path.dirname(_YUNET), exist_ok=True)
            urllib.request.urlretrieve(
                "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
                _YUNET)
        self._yunet = cv2.FaceDetectorYN.create(_YUNET, "", (320, 320),
            score_threshold=config.face_detection_confidence, nms_threshold=0.3, top_k=5000)
        self._sp = dlib.shape_predictor(_SP)
        self._fail_count = 0
        self._smoothed_landmarks: Optional[np.ndarray] = None
        self._alpha = config.landmark_ema_alpha
        self._prev_face_bbox: Optional[tuple] = None
        # Head pose smoothing
        self._smoothed_pitch = 0.0
        self._smoothed_yaw = 0.0
        self._smoothed_roll = 0.0
        self._pose_alpha = 0.4
        logger.info("FaceDetector ready (confidence=%.2f, landmark_ema=%.2f)",
                     config.face_detection_confidence, self._alpha)

    def detect(self, frame, timestamp=0.0) -> FaceResult:
        if timestamp <= 0: timestamp = _time.perf_counter()
        r = FaceResult(frame_timestamp=timestamp)
        h, w = frame.shape[:2]

        small = cv2.resize(frame, (320, 320))
        if small.mean() < 5: return r
        try:
            self._yunet.setInputSize((320, 320))
            count, data = self._yunet.detect(small)
        except Exception:
            return r
        if count == 0 or data is None: return r

        face = data[int(np.argmax(data[:,-1]))]
        sx, sy = w/320, h/320
        x1,y1 = max(0,int(face[0]*sx)), max(0,int(face[1]*sy))
        x2,y2 = min(w,int((face[0]+face[2])*sx)), min(h,int((face[1]+face[3])*sy))
        if x2-x1<10 or y2-y1<10: return r

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mx,my = int((x2-x1)*0.15), int((y2-y1)*0.15)

        shape = None
        for rect in [dlib.rectangle(max(0,x1-mx),max(0,y1-my),min(w,x2+mx),min(h,y2+my)),
                      dlib.rectangle(x1,y1,x2,y2)]:
            try:
                shape = self._sp(gray, rect)
                break
            except Exception:
                pass

        if shape is None:
            self._fail_count += 1
            if self._fail_count <= 3:
                logger.warning("dlib FAILED: frame=%dx%d face=(%d,%d,%d,%d)", w,h,x1,y1,x2-x1,y2-y1)
            return r

        self._fail_count = 0
        lm = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)], dtype=np.float64)
        if lm[:,0].max()>w or lm[:,1].max()>h: return r

        # EMA smoothing of landmarks
        if self._smoothed_landmarks is not None and self._smoothed_landmarks.shape == lm.shape:
            prev_center = self._smoothed_landmarks.mean(axis=0)
            curr_center = lm.mean(axis=0)
            dist = np.linalg.norm(curr_center - prev_center)
            bbox_diag = max(x2-x1, y2-y1)
            if dist < bbox_diag * 0.3:
                self._smoothed_landmarks = self._alpha * lm + (1 - self._alpha) * self._smoothed_landmarks
            else:
                self._smoothed_landmarks = lm.copy()
        else:
            self._smoothed_landmarks = lm.copy()

        lm_smooth = self._smoothed_landmarks

        le, re = compute_ear(lm_smooth, LEFT_EYE), compute_ear(lm_smooth, RIGHT_EYE)
        r.detected = True
        r.bbox = (x1,y1,x2-x1,y2-y1)
        r.confidence = float(data[:,-1].max())
        r.landmarks = lm_smooth
        r.left_ear, r.right_ear, r.avg_ear = le, re, (le+re)/2

        # Head pose estimation
        pitch, yaw, roll = estimate_head_pose(lm_smooth, frame.shape)
        self._smoothed_pitch = self._pose_alpha * pitch + (1 - self._pose_alpha) * self._smoothed_pitch
        self._smoothed_yaw = self._pose_alpha * yaw + (1 - self._pose_alpha) * self._smoothed_yaw
        self._smoothed_roll = self._pose_alpha * roll + (1 - self._pose_alpha) * self._smoothed_roll
        r.head_pitch = self._smoothed_pitch
        r.head_yaw = self._smoothed_yaw
        r.head_roll = self._smoothed_roll

        # Yawning
        mouth_top = (lm_smooth[62] + lm_smooth[63]) / 2
        mouth_bot = (lm_smooth[66] + lm_smooth[65]) / 2
        mouth_left = lm_smooth[60]
        mouth_right = lm_smooth[64]
        vertical = np.linalg.norm(mouth_top - mouth_bot)
        horizontal = np.linalg.norm(mouth_left - mouth_right)
        r.yawning_ratio = vertical / max(horizontal, 1e-6)
        r.is_yawning = r.yawning_ratio > self.config.yawning_threshold
        return r

    def draw_landmarks(self, frame, r):
        if not r.detected or r.landmarks is None: return frame
        vis = frame.copy()
        h, w = frame.shape[:2]
        x,y,bw,bh = r.bbox
        cv2.rectangle(vis,(x,y),(x+bw,y+bh),(0,255,0),2)
        for i in range(68):
            cv2.circle(vis,(int(r.landmarks[i,0]),int(r.landmarks[i,1])),1,(200,200,200),-1)
        for idx,c in [(LEFT_EYE,(0,255,255)),(RIGHT_EYE,(255,255,0))]:
            pts=[(int(r.landmarks[i,0]),int(r.landmarks[i,1])) for i in idx]
            for j in range(len(pts)): cv2.line(vis,pts[j],pts[(j+1)%len(pts)],c,2)
            for p in pts: cv2.circle(vis,p,3,c,-1)
        ec = (0,255,0) if r.avg_ear>0.25 else ((0,165,255) if r.avg_ear>0.20 else (0,0,255))
        cv2.putText(vis,f"EAR:{r.avg_ear:.3f}",(10,h-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,ec,2)
        cv2.putText(vis,f"L:{r.left_ear:.3f} R:{r.right_ear:.3f}",(10,h-35),cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,200,200),1)
        return vis

    def close(self):
        self._yunet = None
        self._smoothed_landmarks = None
