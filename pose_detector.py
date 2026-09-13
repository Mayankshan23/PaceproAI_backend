# ============================================================
# PacePro - Universal Multi-View Pose Detection
# ============================================================

import cv2
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class PoseAnalyzer:

    # ========================================================
    # LANDMARKS REQUIRED BY PACEPRO
    # ========================================================

    LANDMARK_IDS = {

        # -------------------------
        # LEFT LEG
        # -------------------------

        "left_hip": 23,
        "left_knee": 25,
        "left_ankle": 27,
        "left_heel": 29,
        "left_foot": 31,

        # -------------------------
        # RIGHT LEG
        # -------------------------

        "right_hip": 24,
        "right_knee": 26,
        "right_ankle": 28,
        "right_heel": 30,
        "right_foot": 32,

        # -------------------------
        # UPPER BODY
        # -------------------------

        "left_shoulder": 11,
        "right_shoulder": 12,

        "left_elbow": 13,
        "right_elbow": 14,

        "left_wrist": 15,
        "right_wrist": 16
    }

    # ========================================================
    # INITIALIZE POSE MODEL
    # ========================================================

    def __init__(
        self,
        model_path="models/pose_landmarker_full.task"
    ):

        self.model_path = model_path

        base_options = python.BaseOptions(
            model_asset_path=self.model_path
        )

        options = vision.PoseLandmarkerOptions(
            base_options=base_options
        )

        self.landmarker = (
            vision.PoseLandmarker.create_from_options(
                options
            )
        )

    # ========================================================
    # EXTRACT SELECTED LANDMARKS FROM ONE FRAME
    # ========================================================

    def _extract_landmarks(self, landmarks):

        frame_data = {}

        for name, landmark_id in self.LANDMARK_IDS.items():

            landmark = landmarks[landmark_id]

            frame_data[name] = {

                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z,

                "visibility": landmark.visibility
            }

        return frame_data

    # ========================================================
    # ANALYZE ONE VIDEO
    # ========================================================

    def analyze(
        self,
        video_path,
        view_type="unknown"
    ):

        # ----------------------------------------------------
        # Validate view type
        # ----------------------------------------------------

        valid_views = {
            "front",
            "side1",
            "side2",
            "unknown"
        }

        if view_type not in valid_views:

            raise ValueError(
                f"Invalid view_type: {view_type}. "
                f"Use front, side1, side2, or unknown."
            )

        # ----------------------------------------------------
        # Open video
        # ----------------------------------------------------

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():

            raise ValueError(
                f"Could not open video: {video_path}"
            )

        # ----------------------------------------------------
        # Video information
        # ----------------------------------------------------

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0 or math.isnan(fps):
            fps = 30.0

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        # Enforce maximum 11 seconds per video view
        duration = total_frames / fps if fps > 0 else 0
        if duration > 11.5:
            cap.release()
            raise ValueError(
                f"Video duration ({duration:.1f}s) exceeds the maximum allowed 11 seconds limit. "
                f"Please upload a delivery stride clip trimmed to <= 11 seconds."
            )

        width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        # ----------------------------------------------------
        # Pose data
        # ----------------------------------------------------

        pose_data = []

        frame_number = 0

        frames_without_pose = []

        # ====================================================
        # READ VIDEO FRAME BY FRAME
        # ====================================================

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            # ------------------------------------------------
            # BGR -> RGB
            # ------------------------------------------------

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # ------------------------------------------------
            # Convert to MediaPipe Image
            # ------------------------------------------------

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # ------------------------------------------------
            # Detect pose
            # ------------------------------------------------

            results = self.landmarker.detect(
                mp_image
            )

            # =================================================
            # PERSON DETECTED
            # =================================================

            if results.pose_landmarks:

                landmarks = results.pose_landmarks[0]

                frame_data = {
                    "frame": frame_number
                }

                selected_landmarks = (
                    self._extract_landmarks(
                        landmarks
                    )
                )

                frame_data.update(
                    selected_landmarks
                )

                pose_data.append(
                    frame_data
                )

            # =================================================
            # NO PERSON DETECTED
            # =================================================

            else:

                frames_without_pose.append(
                    frame_number
                )

            frame_number += 1

        # ====================================================
        # CLEANUP VIDEO
        # ====================================================

        cap.release()

        # ====================================================
        # RETURN STRUCTURED RESULT
        # ====================================================

        result = {

            "view": view_type,

            "video": {
                "path": video_path,
                "fps": fps,
                "total_frames": total_frames,
                "width": width,
                "height": height
            },

            "pose": {
                "pose_data": pose_data,

                "frames_processed": frame_number,

                "frames_with_pose": len(
                    pose_data
                ),

                "frames_without_pose": len(
                    frames_without_pose
                ),

                "missing_pose_frames":
                    frames_without_pose
            }
        }

        return result

    # ========================================================
    # CLOSE MEDIAPIPE MODEL
    # ========================================================

    def close(self):

        if self.landmarker is not None:

            self.landmarker.close()

            self.landmarker = None

    # ========================================================
    # CONTEXT MANAGER SUPPORT
    # ========================================================

    def __enter__(self):

        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback
    ):

        self.close()
