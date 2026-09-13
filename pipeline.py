"""
PacePro - End-to-End Pipeline

    Video(s)
      -> PoseAnalyzer          (existing module, untouched)
      -> DeliveryAnalyzer      (existing module, completed)
      -> BiomechanicsCalculator
      -> ViewFusion
      -> BowlingLLMAnalyzer
      -> Final PacePro Report (JSON-serializable)

Design choices:
  - Each view is processed independently and wrapped in its own
    try/except. One bad video (corrupt file, no pose detected, etc.)
    does not crash the whole run -- it's recorded as a failed view
    and fusion proceeds with whatever views succeeded.
  - The PoseAnalyzer model is loaded once and reused across views
    (loading a mediapipe model per video would be wasteful), and is
    always closed in a `finally` block.
  - The LLM step is entirely optional and failure-tolerant: if it's
    unavailable, the pipeline still returns all the computed
    measurements, just without narrative text.
"""

import json
from pose_detector import PoseAnalyzer
from delivery_analyzer import DeliveryAnalyzer
from biomechanics import BiomechanicsCalculator
from view_fusion import ViewFusion
from llm_analyzer import BowlingLLMAnalyzer


class PaceProPipeline:

    def __init__(self, pose_model_path="models/pose_landmarker_full.task"):
        self.pose_model_path = pose_model_path
        self.delivery_analyzer = DeliveryAnalyzer()
        self.biomechanics = BiomechanicsCalculator()
        self.view_fusion = ViewFusion()
        self.llm_analyzer = BowlingLLMAnalyzer()

    def analyze_bowling(self, video_paths, run_llm=True, bowling_arm_hint=None):
        """
        video_paths: dict like
            {"front": "path/to/front.mp4", "side1": "...", "side2": "..."}
        Any subset of front/side1/side2 is fine -- missing views are
        just handled gracefully, never invented.

        Returns a fully JSON-serializable report dict.
        """

        per_view_raw = {}       # DeliveryAnalyzer output per view (for debugging/transparency)
        per_view_pose = {}      # Landmark pose data per frame for video annotation
        biomechanics_by_view = {}  # BiomechanicsCalculator output per view
        view_errors = {}

        pose_analyzer = None

        try:
            pose_analyzer = PoseAnalyzer(model_path=self.pose_model_path)

            for view_name, video_path in video_paths.items():
                try:
                    pose_result = pose_analyzer.analyze(video_path, view_type=view_name)
                    delivery_result = self.delivery_analyzer.analyze_view(
                        pose_result, bowling_arm_hint=bowling_arm_hint
                    )
                    per_view_raw[view_name] = delivery_result
                    per_view_pose[view_name] = delivery_result.get("smoothed_pose_data", pose_result.get("pose", {}).get("pose_data", []))

                    biomechanics_by_view[view_name] = (
                        self.biomechanics.analyze_view_biomechanics(delivery_result)
                    )

                except Exception as exc:
                    # A failure on one view must not take down the others.
                    view_errors[view_name] = str(exc)
                    biomechanics_by_view[view_name] = None

        finally:
            if pose_analyzer is not None:
                pose_analyzer.close()

        fused = self.view_fusion.fuse(biomechanics_by_view)

        final_measurements = self._build_final_measurements(
            per_view_raw, fused
        )

        llm_report = None
        if run_llm and fused["views_used"]:
            llm_report = self.llm_analyzer.generate_report(final_measurements)

        return {
            "pipeline_status": "ok" if fused["views_used"] else "no_usable_views",
            "views_requested": list(video_paths.keys()),
            "views_used": fused["views_used"],
            "views_missing_or_failed": fused["views_missing_or_failed"],
            "view_errors": view_errors,
            "final_measurements": final_measurements,
            "per_view_raw": per_view_raw,
            "per_view_pose": per_view_pose,
            "llm_report": llm_report,
        }

    def _build_final_measurements(self, per_view_raw, fused):
        """
        Assembles the structured JSON that gets handed to the LLM
        (and returned to the caller): per-view BFC/FFC/release event
        frames (numeric facts only, not narrative), plus the fused
        cross-view measurements.
        """

        events = {}

        for view_name, result in per_view_raw.items():
            if not result.get("success"):
                events[view_name] = {"success": False, "reason": result.get("reason")}
                continue

            events[view_name] = {
                "success": True,
                "bfc": {"frame": result["bfc"]["frame"], "foot": result["bfc"]["foot"]},
                "ffc": {"frame": result["ffc"]["frame"], "foot": result["ffc"]["foot"]},
                "bowling_arm": result["bowling_arm"],
                "release_frame": result["release"]["frame"] if result["release"] else None,
            }

        return {
            "per_view_events": events,
            "bowling_arm": fused["bowling_arm"],
            "view_fusion": {
                "views_used": fused["views_used"],
                "views_missing_or_failed": fused["views_missing_or_failed"],
                "measurements": fused["measurements"],
            },
        }


def to_json_safe(obj):
    """Utility for callers (e.g. main.py) that want to be extra sure
    the report serializes cleanly -- round-trips through json with a
    numeric fallback for any stray numpy scalar."""
    return json.loads(json.dumps(obj, default=lambda o: float(o)))
