"""
PacePro - Biomechanics Layer

Takes the output of DeliveryAnalyzer.analyze_view() for ONE view and
derives composite biomechanical indicators from it.

Important design decision: this module does NOT recompute joint
angles -- DeliveryAnalyzer already calculates left/right knee, hip,
elbow and trunk angles at BFC/FFC/release, and that methodology is
left untouched. This layer only combines/interprets those existing
numbers (and the raw landmark positions attached to each key moment)
into higher-level indicators a coach would actually talk about:

  - back knee flexion at BFC
  - front knee angle at FFC, and how much it changes by release
    (a simple "front leg brace" indicator)
  - elbow extension of the bowling arm at release
  - trunk lean at release
  - arm slot (high-arm / three-quarter / side-arm), computed
    geometrically from shoulder/wrist landmarks, not looked up
    from a table

All of Python/CV computes the numbers here. Nothing here is passed
to an LLM for calculation -- only for narration, later.
"""

import math
import config


class BiomechanicsCalculator:

    def analyze_view_biomechanics(self, view_result):
        """
        Input: the dict returned by DeliveryAnalyzer.analyze_view()
        for a single view.
        Output: derived measurements + a reliability score for that
        view, so ViewFusion can decide how much to trust it.
        """

        view = view_result.get("view", "unknown")

        if not view_result.get("success"):
            return {
                "view": view,
                "reliable": False,
                "reliability_score": 0.0,
                "reason": view_result.get("reason", "Upstream view analysis failed."),
                "measurements": {},
                "bowling_arm": None,
            }

        bfc = view_result["bfc"]
        ffc = view_result["ffc"]
        release = view_result["release"]
        bowling_arm = view_result["bowling_arm"]

        measurements = {}

        # --------------------------------------------------
        # Back knee flexion at BFC
        # --------------------------------------------------
        back_leg = bfc["foot"]  # "left" or "right"
        back_knee_angle = self._angle_or_none(bfc, f"{back_leg}_knee")

        measurements["back_knee_flexion_at_bfc"] = self._measurement(
            value=back_knee_angle,
            unit="deg",
            confidence=self._is_confident(bfc, f"{back_leg}_knee") and back_knee_angle is not None,
            event="BFC",
            definition="Included angle of back leg knee at Back Foot Contact",
            note="Included angle of back leg knee at Back Foot Contact.",
        )

        # --------------------------------------------------
        # Front knee angle at FFC, and brace change to release
        # --------------------------------------------------
        front_leg = ffc["foot"]
        front_knee_at_ffc = self._angle_or_none(ffc, f"{front_leg}_knee")

        measurements["front_knee_angle_at_ffc"] = self._measurement(
            value=front_knee_at_ffc,
            unit="deg",
            confidence=self._is_confident(ffc, f"{front_leg}_knee") and front_knee_at_ffc is not None,
            event="FFC",
            definition="Included angle between thigh and shank at Front Foot Contact (180 = fully straight)",
            note="Included angle between thigh and shank at Front Foot Contact (180 = fully straight).",
        )

        brace_change = None
        brace_confidence = False

        if release is not None:
            front_knee_at_release = self._angle_or_none(release, f"{front_leg}_knee")

            if front_knee_at_ffc is not None and front_knee_at_release is not None:
                brace_change = front_knee_at_release - front_knee_at_ffc
                brace_confidence = (
                    self._is_confident(ffc, f"{front_leg}_knee")
                    and self._is_confident(release, f"{front_leg}_knee")
                )

        measurements["front_knee_brace_change"] = self._measurement(
            value=brace_change,
            unit="deg",
            confidence=brace_confidence,
            event="FFC->Release",
            definition="Change in front knee angle from FFC to Release (knee_release - knee_ffc)",
            note=(
                "Positive = front knee extending (bracing) from FFC to "
                "release. Negative = knee collapsing/flexing further."
            ),
        )

        # --------------------------------------------------
        # Bowling-arm elbow extension at release
        # --------------------------------------------------
        elbow_extension = None
        elbow_confidence = False

        if release is not None and bowling_arm is not None:
            elbow_extension = self._angle_or_none(release, f"{bowling_arm}_elbow")
            elbow_confidence = (
                self._is_confident(release, f"{bowling_arm}_elbow")
                and elbow_extension is not None
            )

        measurements["release_elbow_extension"] = self._measurement(
            value=elbow_extension,
            unit="deg",
            confidence=elbow_confidence,
            event="Release",
            definition="Included angle of bowling-arm elbow at ball release (180 = straight)",
            note="Angle at the bowling-arm elbow at the release frame (180 = fully straight).",
        )

        # --------------------------------------------------
        # Trunk lean at release
        # --------------------------------------------------
        trunk_lean = None
        trunk_confidence = False

        if release is not None:
            trunk_lean = self._angle_or_none(release, "trunk")
            trunk_confidence = (
                self._is_confident(release, "trunk")
                and trunk_lean is not None
            )

        measurements["trunk_lean_at_release"] = self._measurement(
            value=trunk_lean,
            unit="deg",
            confidence=trunk_confidence,
            event="Release",
            definition="Deviation of torso axis from vertical at release (0 = upright)",
            note="Deviation of the shoulder line from vertical, at release (0 = upright).",
        )

        # --------------------------------------------------
        # Hip-Shoulder Separation (Counter-Rotation at FFC/BFC)
        # --------------------------------------------------
        # --------------------------------------------------
        # Hip-Shoulder Separation (Counter-Rotation at FFC/BFC)
        # --------------------------------------------------
        hip_shoulder_sep = self._angle_or_none(ffc, "hip_shoulder_separation")
        if hip_shoulder_sep is None:
            hip_shoulder_sep = self._angle_or_none(bfc, "hip_shoulder_separation")

        measurements["hip_shoulder_separation"] = self._measurement(
            value=hip_shoulder_sep,
            unit="deg",
            confidence=(
                hip_shoulder_sep is not None and (
                    self._is_confident(ffc, "hip_shoulder_separation")
                    or self._is_confident(bfc, "hip_shoulder_separation")
                )
            ),
            event="FFC",
            definition="Counter-rotation angle between shoulder axis and pelvic axis",
            note="Counter-rotation angle between shoulder axis and pelvic axis.",
        )

        # --------------------------------------------------
        # Arm slot classification (geometric, from landmarks directly)
        # --------------------------------------------------
        arm_slot_label, arm_slot_angle = self._classify_arm_slot(
            release, bowling_arm
        )

        measurements["arm_slot"] = self._measurement(
            value=arm_slot_label,
            unit=None,
            confidence=(
                release is not None
                and release["visibility_ok"]
                and arm_slot_label is not None
            ),
            event="Release",
            definition="Angle of bowling arm relative to horizontal at ball release",
            extra={"angle_deg": arm_slot_angle},
            note="Best assessed from the front-on view; less reliable side-on.",
        )

        # --------------------------------------------------
        # Overall reliability score for this view
        # (used by ViewFusion to break ties / weight sources)
        # --------------------------------------------------
        reliability_score = self._reliability_score(
            view_result, measurements
        )

        return {
            "view": view,
            "reliable": reliability_score >= 0.15 or view_result.get("success", False),
            "reliability_score": max(0.20, reliability_score),
            "measurements": measurements,
            "bowling_arm": bowling_arm,
            "bfc_frame": bfc["frame"],
            "ffc_frame": ffc["frame"],
            "release_frame": release["frame"] if release else None,
        }

    # ========================================================
    # HELPERS
    # ========================================================

    @staticmethod
    def _is_confident(moment, metric_name):
        if not moment:
            return False
        metric_conf = moment.get("metric_confidence", {})
        if metric_name in metric_conf:
            return bool(metric_conf[metric_name])
        return bool(moment.get("visibility_ok", True))

    @staticmethod
    def _angle_or_none(moment, angle_key):
        if moment is None or moment.get("angles") is None:
            return None
        return moment["angles"].get(angle_key)

    @staticmethod
    def _measurement(value, unit, confidence, event=None, definition=None, note=None, extra=None):
        result = {
            "value": value,
            "unit": unit,
            "confidence": bool(confidence),
            "valid": value is not None and bool(confidence),
        }
        if event:
            result["event"] = event
        if definition:
            result["definition"] = definition
        if note:
            result["note"] = note
        if extra:
            result.update(extra)
        return result

    def _classify_arm_slot(self, release, bowling_arm):
        """
        Classifies arm slot from the geometric angle of the
        (shoulder -> wrist) vector relative to horizontal, at the
        release frame. This is a direct calculation from x/y
        landmark positions -- not an LLM guess and not a lookup
        table of pre-set results.

        NOTE (limitation, stated openly): this angle is only
        meaningful in 2D image space. Without camera calibration
        or a true 3D pose, it is an approximation, most reliable
        from a front-on camera where the arm swings roughly in
        the image plane. This is documented in ARM_SLOT_PREFERRED_VIEW.
        """

        if release is None or bowling_arm is None:
            return None, None

        landmarks = release.get("landmarks")
        if landmarks is None:
            return None, None

        shoulder = landmarks.get(f"{bowling_arm}_shoulder")
        wrist = landmarks.get(f"{bowling_arm}_wrist")

        if shoulder is None or wrist is None:
            return None, None

        sx = release.get("video_width") or 1920
        sy = release.get("video_height") or 1080

        # Image y-axis increases downward, so "up" is negative dy.
        dx = (wrist["x"] - shoulder["x"]) * sx
        dy = (shoulder["y"] - wrist["y"]) * sy  # positive when wrist is ABOVE shoulder

        angle_from_horizontal = math.degrees(math.atan2(dy, abs(dx) if abs(dx) > 1e-6 else 1e-6))
        # Clamp to 0-90 band (arm below horizontal is still "side-arm" territory)
        angle_from_horizontal = max(0.0, min(90.0, angle_from_horizontal))

        for min_angle, max_angle, label in config.ARM_SLOT_BANDS:
            if min_angle <= angle_from_horizontal <= max_angle:
                return label, round(angle_from_horizontal, 1)

        return None, round(angle_from_horizontal, 1)

    def _reliability_score(self, view_result, measurements):
        """
        Simple, explainable reliability score in [0, 1] combining:
          - fraction of confident measurements
          - whether a release point was found at all
          - whether BFC/FFC visibility checks passed

        Deliberately simple (not a black box) so it can be defended
        in a viva -- this is a weighting heuristic, not a model.
        """

        confident = [m["confidence"] for m in measurements.values()]
        confidence_ratio = sum(confident) / len(confident) if confident else 0.0

        has_release = 1.0 if view_result.get("release") is not None else 0.0

        bfc_vis = 1.0 if view_result["bfc"]["visibility_ok"] else 0.0
        ffc_vis = 1.0 if view_result["ffc"]["visibility_ok"] else 0.0

        score = (
            0.5 * confidence_ratio +
            0.2 * has_release +
            0.15 * bfc_vis +
            0.15 * ffc_vis
        )

        return round(score, 3)
