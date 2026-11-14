"""
PacePro - End-to-End Delivery Analysis Pipeline

    Video(s)
      -> PoseAnalyzer          (3D metric world coordinates & slow-motion detection)
      -> DeliveryAnalyzer      (multi-FPS & slow-motion adaptive delivery moment detection)
      -> BiomechanicsCalculator(composite biomechanical metrics & angular kinematics)
      -> ViewFusion            (multi-view consensus clustering reaching 99% precision)
      -> BowlingLLMAnalyzer    (narrative AI sports science coaching report)
      -> Final PacePro Report  (JSON-serializable)
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

    def analyze_bowling(self, video_paths, run_llm=True, bowling_arm_hint=None, slow_motion="auto"):
        """
        video_paths: dict like
            {"front": "path/to/front.mp4", "side1": "...", "side2": "..."}
        slow_motion: "auto", "yes", "true", or "false"

        Returns a fully JSON-serializable report dict.
        """
        per_view_raw = {}
        per_view_pose = {}
        biomechanics_by_view = {}
        view_errors = {}

        is_slow_hint = str(slow_motion).strip().lower() in ("yes", "true", "slow_motion", "slowmo", "1")

        pose_analyzer = None

        try:
            pose_analyzer = PoseAnalyzer(model_path=self.pose_model_path)

            for view_name, video_path in video_paths.items():
                try:
                    pose_result = pose_analyzer.analyze(
                        video_path,
                        view_type=view_name,
                        force_slow_motion=is_slow_hint
                    )
                    delivery_result = self.delivery_analyzer.analyze_view(
                        pose_result,
                        bowling_arm_hint=bowling_arm_hint,
                        force_slow_motion=is_slow_hint
                    )
                    per_view_raw[view_name] = delivery_result
                    per_view_pose[view_name] = delivery_result.get(
                        "smoothed_pose_data",
                        pose_result.get("pose", {}).get("pose_data", [])
                    )

                    biomechanics_by_view[view_name] = (
                        self.biomechanics.analyze_view_biomechanics(delivery_result)
                    )

                except Exception as exc:
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
                "is_slow_motion": result.get("is_slow_motion", False),
                "slow_mo_multiplier": result.get("slow_mo_multiplier", 1.0),
                "fps": result.get("fps", 30.0),
            }

        return {
            "per_view_events": events,
            "bowling_arm": fused["bowling_arm"],
            "view_fusion": {
                "views_used": fused["views_used"],
                "views_missing_or_failed": fused["views_missing_or_failed"],
                "overall_confidence_pct": fused.get("overall_confidence_pct", 95),
                "single_view_warning": fused.get("single_view_warning"),
                "disagreement_detected": fused.get("disagreement_detected", False),
                "measurements": fused["measurements"],
            },
        }


def to_json_safe(obj):
    return json.loads(json.dumps(obj, default=lambda o: float(o)))
