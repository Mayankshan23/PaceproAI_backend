"""
PacePro - Multi-View Fusion & Consensus Clustering Layer
=========================================================

Combines per-view biomechanics results (front / side1 / side2)
into ONE final set of measurements with mathematically consistent
confidence calculation, consensus clustering, outlier filtering,
and transparent disagreement tracking.

Rules:
  1. Never invent a value for a measurement no view produced.
  2. Single View:
     - Confidence is STRICTLY CAPPED at < 45% (e.g. 35-42%).
     - Emits prominent single-view low-confidence warning.
  3. Two Views:
     - If agreeing (spread <= tolerance): fuses both with high confidence (75-90%).
     - If strongly disagreeing (spread > tolerance): avoids blind averaging,
       uses biomechanically preferred view, flags view disagreement, and lowers confidence.
  4. Three Views:
     - If all three agree within tolerance: all three contribute (confidence 88-97%).
     - If two views agree and one is an outlier: two-close-values rule clusters
       the agreeing views and explicitly rejects the outlier.
     - If all three diverge: flags strong disagreement and lowers confidence.
  5. Retains candidate values per view (front, side1, side2) for full UI transparency.
"""

import math
import config
from typing import Dict, Any, List, Optional, Tuple


class ViewFusion:

    def fuse(self, biomechanics_by_view: Dict[str, Any]) -> Dict[str, Any]:
        """
        Input: dict like
            {
              "front": {...biomechanics result or None...},
              "side1": {...},
              "side2": {...},
            }
        Views that failed pose detection or delivery analysis should be None.
        """
        views_used = []
        views_missing_or_failed = []
        usable = {}

        for view_name, result in biomechanics_by_view.items():
            if result is None or not result.get("reliable", False):
                views_missing_or_failed.append(view_name)
                continue

            usable[view_name] = result
            views_used.append(view_name)

        # Collect union of measurement names seen across usable views
        measurement_names = [
            "back_knee_flexion_at_bfc",
            "front_knee_angle_at_ffc",
            "front_knee_brace_change",
            "release_elbow_extension",
            "trunk_lean_at_release",
            "hip_shoulder_separation",
            "arm_slot"
        ]
        # Include any extra measurement present in usable views
        for result in usable.values():
            for k in result.get("measurements", {}).keys():
                if k not in measurement_names:
                    measurement_names.append(k)

        fused_measurements = {}
        all_confidences = []
        disagreement_detected = False

        for name in measurement_names:
            fused_item = self._fuse_single_measurement(name, usable)
            fused_measurements[name] = fused_item
            if fused_item.get("confidence_pct") is not None and fused_item["confidence_pct"] > 0:
                all_confidences.append(fused_item["confidence_pct"])
            if fused_item.get("status") == "view_disagreement":
                disagreement_detected = True

        bowling_arm_result = self._fuse_bowling_arm(usable)

        # Overall Session Confidence & Single-View Guard
        num_views = len(views_used)
        if num_views == 0:
            overall_confidence_pct = 0
            single_view_warning = None
        elif num_views == 1:
            # MANDATORY: Single view confidence MUST be strictly < 45%
            avg_cand = sum(all_confidences) / len(all_confidences) if all_confidences else 38
            overall_confidence_pct = max(25, min(42, int(round(avg_cand))))
            single_view_warning = (
                "⚠️ LOW CONFIDENCE — SINGLE VIEW ANALYSIS: Single camera perspective lacks depth parallax. "
                "Upload Front + Side View 1 + Side View 2 for multi-view fusion and high confidence."
            )
        else:
            single_view_warning = None
            if all_confidences:
                overall_confidence_pct = int(round(sum(all_confidences) / len(all_confidences)))
            else:
                overall_confidence_pct = 70

        return {
            "views_used": views_used,
            "views_missing_or_failed": views_missing_or_failed,
            "num_views": num_views,
            "bowling_arm": bowling_arm_result,
            "measurements": fused_measurements,
            "overall_confidence_pct": overall_confidence_pct,
            "single_view_warning": single_view_warning,
            "disagreement_detected": disagreement_detected,
        }

    # ========================================================
    # FUSE ONE MEASUREMENT ACROSS AVAILABLE VIEWS
    # ========================================================

    def _fuse_single_measurement(self, name: str, usable_views: Dict[str, Any]) -> Dict[str, Any]:
        candidates = []  # (view_name, value, unit, cand_conf, rel_score, event, definition)
        candidates_by_view = {}

        for view_name, result in usable_views.items():
            measurement = result.get("measurements", {}).get(name)
            if measurement is None:
                continue
            val = measurement.get("value")
            unit = measurement.get("unit")
            conf = measurement.get("confidence", False)
            event = measurement.get("event")
            definition = measurement.get("definition")
            rel_score = result.get("reliability_score", 0.7)

            if val is not None:
                candidates_by_view[view_name] = val
                if conf:
                    candidates.append((view_name, val, unit, conf, rel_score, event, definition))

        preference_order = config.MEASUREMENT_VIEW_PREFERENCE.get(name, ["side1", "side2", "front"])
        tolerance = getattr(config, "FUSION_AGREEMENT_TOLERANCE_DEG", 8.0)

        # Fallback to any available measurement across views if confident candidate list is empty
        if not candidates and candidates_by_view:
            for pref_v in preference_order:
                if pref_v in candidates_by_view and candidates_by_view[pref_v] is not None:
                    res_m = usable_views[pref_v].get("measurements", {}).get(name, {})
                    candidates.append((
                        pref_v,
                        candidates_by_view[pref_v],
                        res_m.get("unit"),
                        False,
                        0.30,
                        res_m.get("event"),
                        res_m.get("definition")
                    ))
                    break
            if not candidates:
                first_v = next(iter(candidates_by_view.keys()))
                res_m = usable_views[first_v].get("measurements", {}).get(name, {})
                candidates.append((
                    first_v,
                    candidates_by_view[first_v],
                    res_m.get("unit"),
                    False,
                    0.30,
                    res_m.get("event"),
                    res_m.get("definition")
                ))

        # Event & Definition metadata from first available
        event_name = candidates[0][5] if candidates else None
        metric_def = candidates[0][6] if candidates else None
        unit = candidates[0][2] if candidates else ("deg" if "arm_slot" not in name else None)

        # -------------------------------------------------
        # Case 0: No valid candidate
        # -------------------------------------------------
        if not candidates:
            return {
                "value": None,
                "unit": unit,
                "source": None,
                "confidence": "none",
                "confidence_pct": 0,
                "status": "insufficient_data",
                "event": event_name,
                "definition": metric_def,
                "selected_views": [],
                "rejected_outliers": [],
                "candidates": candidates_by_view,
                "front": candidates_by_view.get("front"),
                "side1": candidates_by_view.get("side1"),
                "side2": candidates_by_view.get("side2"),
                "note": "No view produced a confident measurement for this metric.",
            }

        # -------------------------------------------------
        # Non-numeric measurement (e.g. arm_slot string label)
        # -------------------------------------------------
        if not all(isinstance(c[1], (int, float)) for c in candidates):
            return self._fuse_categorical(name, candidates, preference_order, unit, candidates_by_view)

        num_candidates = len(candidates)

        # -------------------------------------------------
        # Case 1: Exactly 1 valid candidate (Single View)
        # -------------------------------------------------
        if num_candidates == 1:
            view_name, val, _, _, rel_score, _, _ = candidates[0]
            # MANDATORY: Single view confidence MUST be strictly < 45%
            cand_conf_pct = max(25, min(42, int(round(rel_score * 42))))
            return {
                "value": round(float(val), 1),
                "unit": unit,
                "source": view_name,
                "confidence": "low",
                "confidence_pct": cand_conf_pct,
                "status": "single_view_low_confidence",
                "event": event_name,
                "definition": metric_def,
                "selected_views": [view_name],
                "rejected_outliers": [],
                "candidates": candidates_by_view,
                "front": candidates_by_view.get("front"),
                "side1": candidates_by_view.get("side1"),
                "side2": candidates_by_view.get("side2"),
                "note": "Single-view measurement (confidence capped < 45%). Upload additional views for multi-view fusion.",
            }

        # -------------------------------------------------
        # Case 2: Exactly 2 valid candidates
        # -------------------------------------------------
        if num_candidates == 2:
            v0_name, v0_val, _, _, w0, _, _ = candidates[0]
            v1_name, v1_val, _, _, w1, _, _ = candidates[1]
            spread = abs(v0_val - v1_val)

            if spread <= tolerance:
                total_w = max(0.1, w0 + w1)
                fused_val = round((v0_val * w0 + v1_val * w1) / total_w, 1)
                # Strong agreement between 2 views: confidence 75% - 90%
                conf_factor = max(0.0, 1.0 - (spread / tolerance))
                conf_pct = int(round(75 + conf_factor * 15))
                return {
                    "value": fused_val,
                    "unit": unit,
                    "source": f"combined({v0_name}+{v1_name})",
                    "confidence": "high",
                    "confidence_pct": conf_pct,
                    "status": "two_view_agreement",
                    "event": event_name,
                    "definition": metric_def,
                    "selected_views": [v0_name, v1_name],
                    "rejected_outliers": [],
                    "candidates": candidates_by_view,
                    "front": candidates_by_view.get("front"),
                    "side1": candidates_by_view.get("side1"),
                    "side2": candidates_by_view.get("side2"),
                    "note": f"2 views agreed within {tolerance}° tolerance (spread = {spread:.1f}°).",
                }
            else:
                # Strong disagreement: Do NOT blindly average!
                # Choose the biomechanically preferred view, record disagreement
                chosen = self._pick_by_preference(candidates, preference_order)
                other = [c for c in candidates if c[0] != chosen[0]][0]
                # Lower confidence significantly
                conf_pct = min(48, max(28, int(round(50 - (spread - tolerance) * 1.5))))
                return {
                    "value": round(float(chosen[1]), 1),
                    "unit": unit,
                    "source": chosen[0],
                    "confidence": "low",
                    "confidence_pct": conf_pct,
                    "status": "view_disagreement",
                    "event": event_name,
                    "definition": metric_def,
                    "selected_views": [chosen[0]],
                    "rejected_outliers": [other[0]],
                    "candidates": candidates_by_view,
                    "front": candidates_by_view.get("front"),
                    "side1": candidates_by_view.get("side1"),
                    "side2": candidates_by_view.get("side2"),
                    "note": f"⚠️ VIEW DISAGREEMENT: Views disagreed by {spread:.1f}° (> {tolerance}°). Used preferred view {chosen[0]}.",
                }

        # -------------------------------------------------
        # Case 3: 3 valid candidates (Full 3-View Multi-View Fusion)
        # -------------------------------------------------
        v = [float(c[1]) for c in candidates]
        names = [c[0] for c in candidates]
        weights = [max(0.1, float(c[4])) for c in candidates]

        d01 = abs(v[0] - v[1])
        d12 = abs(v[1] - v[2])
        d02 = abs(v[0] - v[2])
        max_spread = max(d01, d12, d02)

        # Subcase 3A: All 3 views agree within tolerance
        if max_spread <= tolerance:
            fused_val = round(sum(val * w for val, w in zip(v, weights)) / sum(weights), 1)
            conf_factor = max(0.0, 1.0 - (max_spread / tolerance))
            conf_pct = int(round(88 + conf_factor * 9))  # 88% - 97%
            return {
                "value": fused_val,
                "unit": unit,
                "source": "combined(" + "+".join(names) + ")",
                "confidence": "very_high",
                "confidence_pct": conf_pct,
                "status": "three_view_consensus",
                "event": event_name,
                "definition": metric_def,
                "selected_views": names,
                "rejected_outliers": [],
                "candidates": candidates_by_view,
                "front": candidates_by_view.get("front"),
                "side1": candidates_by_view.get("side1"),
                "side2": candidates_by_view.get("side2"),
                "note": f"Full 3-view consensus (all views agree within {tolerance}°, spread = {max_spread:.1f}°).",
            }

        # Subcase 3B: Two-Close-Values Rule (Consensus Cluster & Outlier Rejection)
        # Candidate 2 is outlier
        if d01 <= tolerance and d02 > tolerance and d12 > tolerance:
            fused_val = round((v[0] * weights[0] + v[1] * weights[1]) / (weights[0] + weights[1]), 1)
            conf_pct = int(round(80 + (1.0 - (d01 / tolerance)) * 10))
            return {
                "value": fused_val,
                "unit": unit,
                "source": f"consensus({names[0]}+{names[1]})",
                "confidence": "high",
                "confidence_pct": conf_pct,
                "status": "consensus_cluster",
                "event": event_name,
                "definition": metric_def,
                "selected_views": [names[0], names[1]],
                "rejected_outliers": [names[2]],
                "candidates": candidates_by_view,
                "front": candidates_by_view.get("front"),
                "side1": candidates_by_view.get("side1"),
                "side2": candidates_by_view.get("side2"),
                "note": f"Consensus cluster on {names[0]} & {names[1]} (spread {d01:.1f}°). Rejected outlier {names[2]} ({v[2]:.1f}°).",
            }

        # Candidate 0 is outlier
        if d12 <= tolerance and d01 > tolerance and d02 > tolerance:
            fused_val = round((v[1] * weights[1] + v[2] * weights[2]) / (weights[1] + weights[2]), 1)
            conf_pct = int(round(80 + (1.0 - (d12 / tolerance)) * 10))
            return {
                "value": fused_val,
                "unit": unit,
                "source": f"consensus({names[1]}+{names[2]})",
                "confidence": "high",
                "confidence_pct": conf_pct,
                "status": "consensus_cluster",
                "event": event_name,
                "definition": metric_def,
                "selected_views": [names[1], names[2]],
                "rejected_outliers": [names[0]],
                "candidates": candidates_by_view,
                "front": candidates_by_view.get("front"),
                "side1": candidates_by_view.get("side1"),
                "side2": candidates_by_view.get("side2"),
                "note": f"Consensus cluster on {names[1]} & {names[2]} (spread {d12:.1f}°). Rejected outlier {names[0]} ({v[0]:.1f}°).",
            }

        # Candidate 1 is outlier
        if d02 <= tolerance and d01 > tolerance and d12 > tolerance:
            fused_val = round((v[0] * weights[0] + v[2] * weights[2]) / (weights[0] + weights[2]), 1)
            conf_pct = int(round(80 + (1.0 - (d02 / tolerance)) * 10))
            return {
                "value": fused_val,
                "unit": unit,
                "source": f"consensus({names[0]}+{names[2]})",
                "confidence": "high",
                "confidence_pct": conf_pct,
                "status": "consensus_cluster",
                "event": event_name,
                "definition": metric_def,
                "selected_views": [names[0], names[2]],
                "rejected_outliers": [names[1]],
                "candidates": candidates_by_view,
                "front": candidates_by_view.get("front"),
                "side1": candidates_by_view.get("side1"),
                "side2": candidates_by_view.get("side2"),
                "note": f"Consensus cluster on {names[0]} & {names[2]} (spread {d02:.1f}°). Rejected outlier {names[1]} ({v[1]:.1f}°).",
            }

        # Subcase 3C: All 3 disagree beyond tolerance
        chosen = self._pick_by_preference(candidates, preference_order)
        others = [c[0] for c in candidates if c[0] != chosen[0]]
        return {
            "value": round(float(chosen[1]), 1),
            "unit": unit,
            "source": chosen[0],
            "confidence": "low",
            "confidence_pct": 38,
            "status": "view_disagreement",
            "event": event_name,
            "definition": metric_def,
            "selected_views": [chosen[0]],
            "rejected_outliers": others,
            "candidates": candidates_by_view,
            "front": candidates_by_view.get("front"),
            "side1": candidates_by_view.get("side1"),
            "side2": candidates_by_view.get("side2"),
            "note": f"⚠️ VIEW DISAGREEMENT: All 3 views diverged beyond {tolerance}°. Selected preferred view {chosen[0]}.",
        }

    # ========================================================
    # CATEGORICAL MEASUREMENT FUSION (e.g. arm_slot label)
    # ========================================================

    def _fuse_categorical(self, name: str, candidates: list, preference_order: list, unit, candidates_by_view: dict) -> dict:
        labels = [c[1] for c in candidates]
        num_candidates = len(candidates)
        event_name = candidates[0][5] if candidates else "Release"
        metric_def = candidates[0][6] if candidates else "Arm slot classification"

        if num_candidates == 1:
            view_name, val, _, _, rel_score, _, _ = candidates[0]
            return {
                "value": val,
                "unit": unit,
                "source": view_name,
                "confidence": "low",
                "confidence_pct": max(25, min(42, int(round(rel_score * 42)))),
                "status": "single_view_low_confidence",
                "event": event_name,
                "definition": metric_def,
                "selected_views": [view_name],
                "rejected_outliers": [],
                "candidates": candidates_by_view,
                "front": candidates_by_view.get("front"),
                "side1": candidates_by_view.get("side1"),
                "side2": candidates_by_view.get("side2"),
                "note": "Single-view observation (confidence capped < 45%).",
            }

        if len(set(labels)) == 1:
            return {
                "value": labels[0],
                "unit": unit,
                "source": "combined(" + "+".join(c[0] for c in candidates) + ")",
                "confidence": "high",
                "confidence_pct": 90,
                "status": "agreement",
                "event": event_name,
                "definition": metric_def,
                "selected_views": [c[0] for c in candidates],
                "rejected_outliers": [],
                "candidates": candidates_by_view,
                "front": candidates_by_view.get("front"),
                "side1": candidates_by_view.get("side1"),
                "side2": candidates_by_view.get("side2"),
                "note": "All views agreed on classification.",
            }

        chosen = self._pick_by_preference(candidates, preference_order)
        others = [c[0] for c in candidates if c[0] != chosen[0]]
        return {
            "value": chosen[1],
            "unit": unit,
            "source": chosen[0],
            "confidence": "medium",
            "confidence_pct": 60,
            "status": "view_disagreement",
            "event": event_name,
            "definition": metric_def,
            "selected_views": [chosen[0]],
            "rejected_outliers": others,
            "candidates": candidates_by_view,
            "front": candidates_by_view.get("front"),
            "side1": candidates_by_view.get("side1"),
            "side2": candidates_by_view.get("side2"),
            "note": f"Views disagreed on category. Used biomechanically preferred view {chosen[0]}.",
        }

    @staticmethod
    def _pick_by_preference(candidates, preference_order):
        candidates_by_view = {c[0]: c for c in candidates}
        for preferred_view in preference_order:
            if preferred_view in candidates_by_view:
                return candidates_by_view[preferred_view]
        return candidates[0]

    # ========================================================
    # FUSE BOWLING ARM ACROSS VIEWS
    # ========================================================

    def _fuse_bowling_arm(self, usable_views: Dict[str, Any]) -> Dict[str, Any]:
        votes = {}
        for view_name, result in usable_views.items():
            arm = result.get("bowling_arm")
            if arm:
                votes.setdefault(arm, []).append(view_name)

        if not votes:
            return {
                "value": None,
                "source": None,
                "confidence": "none",
                "confidence_pct": 0,
                "note": "No view could determine bowling arm.",
            }

        if len(votes) == 1:
            arm, views = next(iter(votes.items()))
            conf_pct = 95 if len(views) > 1 else 38
            return {
                "value": arm,
                "source": "combined(" + "+".join(views) + ")" if len(views) > 1 else views[0],
                "confidence": "high" if len(views) > 1 else "low",
                "confidence_pct": conf_pct,
                "note": None if len(views) > 1 else "Single-view observation (confidence < 45%).",
            }

        best_arm, best_views = max(votes.items(), key=lambda kv: len(kv[1]))
        other = {a: v for a, v in votes.items() if a != best_arm}
        return {
            "value": best_arm,
            "source": "+".join(best_views),
            "confidence": "medium",
            "confidence_pct": 65,
            "note": f"Majority vote ({best_arm}) chosen. Disagreement from: {other}",
        }


# =====================================================================
# SELF-TEST & VERIFICATION RUNNER
# =====================================================================

if __name__ == "__main__":
    import sys
    import json

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 70)
    print("  PacePro AI - Multi-View Fusion & Consensus Clustering Test")
    print("=" * 70)

    fusion = ViewFusion()

    # -------------------------------------------------------------
    # Test Case 1: Single View (Confidence capped strictly < 45%)
    # -------------------------------------------------------------
    print("\n--- TEST CASE 1: Single View (side1 only) ---")
    single_view_input = {
        "front": None,
        "side1": {
            "reliable": True,
            "reliability_score": 0.88,
            "bowling_arm": "right",
            "measurements": {
                "front_knee_angle_at_ffc": {
                    "value": 162.5,
                    "unit": "deg",
                    "confidence": True,
                    "event": "FFC",
                    "definition": "Angle of front knee at front-foot contact",
                },
                "release_elbow_extension": {
                    "value": 172.0,
                    "unit": "deg",
                    "confidence": True,
                    "event": "Release",
                    "definition": "Release elbow extension angle",
                },
                "arm_slot": {
                    "value": "high_three_quarters",
                    "unit": None,
                    "confidence": True,
                    "event": "Release",
                    "definition": "Bowling arm slot classification",
                },
            },
        },
        "side2": None,
    }
    res1 = fusion.fuse(single_view_input)
    print(f"Views used: {res1['views_used']}")
    print(f"Overall Confidence: {res1['overall_confidence_pct']}% (Capped < 45%: {res1['overall_confidence_pct'] < 45})")
    print(f"Warning: {res1['single_view_warning']}")
    fk1 = res1["measurements"]["front_knee_angle_at_ffc"]
    print(f"Front Knee Angle: {fk1['value']}{fk1['unit']} (source: {fk1['source']}, conf: {fk1['confidence_pct']}%, status: {fk1['status']})")

    # -------------------------------------------------------------
    # Test Case 2: Two Views with Disagreement (> tolerance)
    # -------------------------------------------------------------
    print("\n--- TEST CASE 2: Two Views with Disagreement (side1 vs side2 > 8° spread) ---")
    two_view_disagree = {
        "side1": {
            "reliable": True,
            "reliability_score": 0.85,
            "bowling_arm": "right",
            "measurements": {
                "front_knee_angle_at_ffc": {
                    "value": 160.0,
                    "unit": "deg",
                    "confidence": True,
                    "event": "FFC",
                    "definition": "Angle of front knee at front-foot contact",
                }
            },
        },
        "side2": {
            "reliable": True,
            "reliability_score": 0.82,
            "bowling_arm": "right",
            "measurements": {
                "front_knee_angle_at_ffc": {
                    "value": 174.0,  # 14 deg difference > 8 deg tolerance
                    "unit": "deg",
                    "confidence": True,
                    "event": "FFC",
                    "definition": "Angle of front knee at front-foot contact",
                }
            },
        },
    }
    res2 = fusion.fuse(two_view_disagree)
    fk2 = res2["measurements"]["front_knee_angle_at_ffc"]
    print(f"Views used: {res2['views_used']}")
    print(f"Disagreement detected: {res2['disagreement_detected']}")
    print(f"Front Knee Angle: {fk2['value']}{fk2['unit']} (status: {fk2['status']}, conf: {fk2['confidence_pct']}%)")
    print(f"Candidates preserved: {fk2['candidates']}")
    print(f"Note: {fk2['note']}")

    # -------------------------------------------------------------
    # Test Case 3: Three Views with Outlier Rejection (Two-Close-Values Rule)
    # -------------------------------------------------------------
    print("\n--- TEST CASE 3: Three Views with Outlier Rejection ---")
    three_view_input = {
        "front": {
            "reliable": True,
            "reliability_score": 0.75,
            "bowling_arm": "right",
            "measurements": {
                "front_knee_angle_at_ffc": {
                    "value": 182.0,  # Outlier! front view parallax distortion
                    "unit": "deg",
                    "confidence": True,
                    "event": "FFC",
                    "definition": "Angle of front knee at front-foot contact",
                },
                "arm_slot": {
                    "value": "over_the_top",
                    "unit": None,
                    "confidence": True,
                    "event": "Release",
                    "definition": "Bowling arm slot classification",
                },
            },
        },
        "side1": {
            "reliable": True,
            "reliability_score": 0.90,
            "bowling_arm": "right",
            "measurements": {
                "front_knee_angle_at_ffc": {
                    "value": 162.0,  # Agreeing cluster
                    "unit": "deg",
                    "confidence": True,
                    "event": "FFC",
                    "definition": "Angle of front knee at front-foot contact",
                },
                "arm_slot": {
                    "value": "over_the_top",
                    "unit": None,
                    "confidence": True,
                    "event": "Release",
                    "definition": "Bowling arm slot classification",
                },
            },
        },
        "side2": {
            "reliable": True,
            "reliability_score": 0.92,
            "bowling_arm": "right",
            "measurements": {
                "front_knee_angle_at_ffc": {
                    "value": 164.0,  # Agreeing cluster (|164 - 162| = 2 deg <= 8 deg)
                    "unit": "deg",
                    "confidence": True,
                    "event": "FFC",
                    "definition": "Angle of front knee at front-foot contact",
                },
                "arm_slot": {
                    "value": "over_the_top",
                    "unit": None,
                    "confidence": True,
                    "event": "Release",
                    "definition": "Bowling arm slot classification",
                },
            },
        },
    }
    res3 = fusion.fuse(three_view_input)
    fk3 = res3["measurements"]["front_knee_angle_at_ffc"]
    print(f"Views used: {res3['views_used']}")
    print(f"Overall Session Confidence: {res3['overall_confidence_pct']}%")
    print(f"Fused Front Knee Value: {fk3['value']}{fk3['unit']}")
    print(f"Selected Views: {fk3['selected_views']}")
    print(f"Rejected Outliers: {fk3['rejected_outliers']}")
    print(f"Confidence: {fk3['confidence_pct']}% (status: {fk3['status']})")
    print(f"Note: {fk3['note']}")
    print("\n" + "=" * 70)
    print("  All Multi-View Fusion tests executed successfully!")
    print("=" * 70)

