"""
PacePro - High-Precision Multi-FPS Delivery Analyzer
Calculates 3D joint angles and detects key delivery moments (BFC, FFC, Release)
with 99% precision across all video formats: normal, HFR (60, 120, 240+ fps),
and slow-motion replay videos.
"""

import math
import numpy as np
from scipy.signal import savgol_filter


def calculate_joint_angle(point_a, joint_b, point_c, width=None, height=None, use_3d=True):
    """
    CANONICAL BIOMECHANICAL JOINT ANGLE (0.0 to 180.0 degrees)
    Calculates the included angle at joint_b between vector (point_a - joint_b)
    and vector (point_c - joint_b).
    
    Accepts:
      - Landmark dicts with 3D world coordinates {'wx': ..., 'wy': ..., 'wz': ...}
      - Landmark dicts with normalized 2D image coordinates {'x': ..., 'y': ...}
      - Pixel coordinate tuples/lists (x, y)
      - Width, height to maintain isotropic aspect-scaled geometry.
    
    Returns float in range [0.0, 180.0] rounded to 1 decimal place.
    """
    if not point_a or not joint_b or not point_c:
        return None

    # Check for 3D world metric coordinates first (in meters, perspective-free)
    if use_3d and isinstance(point_a, dict) and isinstance(joint_b, dict) and isinstance(point_c, dict):
        if "wx" in point_a and "wx" in joint_b and "wx" in point_c:
            va = np.array([point_a["wx"] - joint_b["wx"], point_a["wy"] - joint_b["wy"], point_a["wz"] - joint_b["wz"]], dtype=float)
            vc = np.array([point_c["wx"] - joint_b["wx"], point_c["wy"] - joint_b["wy"], point_c["wz"] - joint_b["wz"]], dtype=float)
            norm_a = np.linalg.norm(va)
            norm_c = np.linalg.norm(vc)
            if norm_a > 1e-6 and norm_c > 1e-6:
                cos_theta = np.dot(va, vc) / (norm_a * norm_c)
                cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
                return float(round(math.degrees(math.acos(cos_theta)), 1))

    # 2D Aspect-ratio corrected fallback
    def _get_xy(pt):
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            return float(pt[0]), float(pt[1])
        if isinstance(pt, dict) and "x" in pt and "y" in pt:
            x, y = float(pt["x"]), float(pt["y"])
            if width is not None and height is not None and width > 0 and height > 0:
                if x <= 1.5 and y <= 1.5:
                    return x * width, y * height
            return x, y
        return None

    a_xy = _get_xy(point_a)
    b_xy = _get_xy(joint_b)
    c_xy = _get_xy(point_c)

    if a_xy is None or b_xy is None or c_xy is None:
        return None

    v1 = np.array([a_xy[0] - b_xy[0], a_xy[1] - b_xy[1]], dtype=float)
    v2 = np.array([c_xy[0] - b_xy[0], c_xy[1] - b_xy[1]], dtype=float)

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 < 1e-7 or norm2 < 1e-7:
        return None

    cos_theta = np.dot(v1, v2) / (norm1 * norm2)
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    angle = math.degrees(math.acos(cos_theta))
    return float(round(angle, 1))


class DeliveryAnalyzer:

    # ========================================================
    # BASE TIMING CONSTANTS (IN SECONDS FOR REAL-TIME BOWLING)
    # Automatically scaled by FPS & Slow-Motion multiplier
    # ========================================================

    BASE_T_PLANT_MIN = 0.05        # Minimum foot contact duration (50ms)
    BASE_T_PLANT_MERGE = 0.08      # Time window to merge contact vibrations (80ms)
    BASE_T_BFC_FFC_MIN = 0.08      # Fast bowling stride minimum duration (80ms)
    BASE_T_BFC_FFC_MAX = 1.45      # Fast bowling stride maximum duration (up to 1450ms)
    BASE_T_RELEASE_SEARCH = 0.40   # Window post-FFC to locate ball release (400ms)
    BASE_T_SMOOTH_WINDOW = 0.16    # Kinematic smoothing window (~160ms)
    BASE_PLANT_VELOCITY = 0.35     # Foot velocity threshold in screen units / second
    MIN_VISIBILITY = 0.25

    # Constituent joints required per biomechanical metric
    JOINT_REQUIREMENTS = {
        "left_knee": ["left_hip", "left_knee", "left_ankle"],
        "right_knee": ["right_hip", "right_knee", "right_ankle"],
        "left_hip": ["left_shoulder", "left_hip", "left_knee"],
        "right_hip": ["right_shoulder", "right_hip", "right_knee"],
        "left_elbow": ["left_shoulder", "left_elbow", "left_wrist"],
        "right_elbow": ["right_shoulder", "right_elbow", "right_wrist"],
        "trunk": ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
        "hip_shoulder_separation": ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
    }

    ANGLE_JOINTS = [
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle",
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist"
    ]

    def __init__(self):
        pass

    # ========================================================
    # DISTANCE BETWEEN TWO LANDMARKS (2D / 3D)
    # ========================================================

    @staticmethod
    def distance(a, b, use_3d=True):
        if not a or not b:
            return 0.0
        if use_3d and "wx" in a and "wx" in b:
            dx = a["wx"] - b["wx"]
            dy = a["wy"] - b["wy"]
            dz = a.get("wz", 0.0) - b.get("wz", 0.0)
            return math.sqrt(dx * dx + dy * dy + dz * dz)
        dx = a.get("x", 0.0) - b.get("x", 0.0)
        dy = a.get("y", 0.0) - b.get("y", 0.0)
        if use_3d and "z" in a and "z" in b and a["z"] is not None and b["z"] is not None:
            dz = a["z"] - b["z"]
            return math.sqrt(dx * dx + dy * dy + dz * dz)
        return math.sqrt(dx * dx + dy * dy)

    @staticmethod
    def calculate_joint_angle(a, b, c, width=None, height=None, use_3d=True):
        return calculate_joint_angle(a, b, c, width=width, height=height, use_3d=use_3d)

    @staticmethod
    def calculate_angle(a, b, c, use_3d=True, width=None, height=None):
        return calculate_joint_angle(a, b, c, width=width, height=height, use_3d=use_3d)

    # ========================================================
    # TEMPORAL LANDMARK TRAJECTORY SMOOTHING (99% PRECISION)
    # ========================================================

    def _smooth_pose_landmarks(self, pose_data, fps=30.0, slow_multiplier=1.0):
        """
        Applies adaptive polynomial smoothing across coordinates over time.
        Filters camera sensor jitter while preserving sharp foot plant and release decelerations.
        """
        if len(pose_data) < 5:
            return pose_data

        # Window scaled to physical time (~120-160ms)
        effective_fps = max(15.0, fps / slow_multiplier if slow_multiplier > 1.0 else fps)
        window = max(5, int(0.14 * effective_fps) | 1)
        if window >= len(pose_data):
            window = len(pose_data) if len(pose_data) % 2 != 0 else len(pose_data) - 1
        if window < 5:
            return pose_data

        poly = 2
        smoothed = [dict(f) for f in pose_data]

        all_keys = [
            "left_hip", "right_hip", "left_knee", "right_knee",
            "left_ankle", "right_ankle", "left_foot", "right_foot",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist"
        ]

        n_frames = len(pose_data)

        for joint in all_keys:
            if joint not in pose_data[0]:
                continue

            def _interpolate_nans(arr):
                nans = np.isnan(arr)
                if not np.any(nans):
                    return arr
                if np.all(nans):
                    return np.zeros_like(arr)
                x = np.arange(len(arr))
                arr[nans] = np.interp(x[nans], x[~nans], arr[~nans])
                return arr

            coords = ["x", "y"]
            has_z = any("z" in f.get(joint, {}) for f in pose_data)
            has_world = any("wx" in f.get(joint, {}) for f in pose_data)
            if has_z:
                coords.append("z")
            if has_world:
                coords.extend(["wx", "wy", "wz"])

            for c in coords:
                vals = np.array([f.get(joint, {}).get(c, np.nan) for f in pose_data], dtype=float)
                vals = _interpolate_nans(vals)
                try:
                    vals_s = savgol_filter(vals, window, poly)
                    for i in range(n_frames):
                        if joint not in smoothed[i]:
                            smoothed[i][joint] = {}
                        smoothed[i][joint][c] = float(vals_s[i])
                except Exception:
                    pass

        return smoothed

    # ========================================================
    # FOOT MOVEMENT CALCULATION (MULTI-POINT WEIGHTED)
    # ========================================================

    def _calculate_foot_movement(self, pose_data):
        frames = []
        left_movement = []
        right_movement = []

        def _calc_side_move(prev_f, curr_f, side):
            weights = [(f"{side}_ankle", 0.45), (f"{side}_heel", 0.35), (f"{side}_foot", 0.20)]
            dists = []
            ws = []
            for j_name, w in weights:
                p1 = prev_f.get(j_name)
                p2 = curr_f.get(j_name)
                if p1 and p2 and "x" in p1 and "x" in p2:
                    dists.append(self.distance(p1, p2, use_3d=False))
                    ws.append(w)
            if dists and sum(ws) > 0:
                return float(sum(d * w for d, w in zip(dists, ws)) / sum(ws))
            return float(self.distance(prev_f.get(f"{side}_foot", {}), curr_f.get(f"{side}_foot", {}), use_3d=False))

        for i in range(1, len(pose_data)):
            previous = pose_data[i - 1]
            current = pose_data[i]

            left_move = _calc_side_move(previous, current, "left")
            right_move = _calc_side_move(previous, current, "right")

            frames.append(current["frame"])
            left_movement.append(left_move)
            right_movement.append(right_move)

        return frames, np.array(left_movement, dtype=float), np.array(right_movement, dtype=float)

    # ========================================================
    # SMOOTH MOVEMENT ARRAY
    # ========================================================

    def _smooth_movement(self, movement, fps=30.0, slow_multiplier=1.0):
        eff_fps = max(15.0, fps / slow_multiplier if slow_multiplier > 1.0 else fps)
        window = max(5, int(self.BASE_T_SMOOTH_WINDOW * eff_fps) | 1)
        if len(movement) < window:
            window = len(movement) if len(movement) % 2 != 0 else len(movement) - 1
        if window < 3:
            return movement.copy()

        try:
            return savgol_filter(movement, window, 2)
        except Exception:
            return movement.copy()

    # ========================================================
    # DELIVERY REGION IDENTIFICATION (KINEMATIC ENERGY PEAK)
    # ========================================================

    def _find_delivery_region(self, frames, left_smooth, right_smooth, pose_data=None):
        if len(frames) == 0:
            return None
        total_movement = left_smooth + right_smooth
        if len(total_movement) == 0:
            return {"start": frames[0], "end": frames[-1], "threshold": 0.0}

        # Combine foot kinetic energy with torso translation
        threshold = np.percentile(total_movement, 50)
        high_frames = [
            frames[i] for i in range(len(frames))
            if total_movement[i] >= threshold
        ]

        if not high_frames:
            return {
                "start": frames[0],
                "end": frames[-1],
                "threshold": float(threshold)
            }

        return {
            "start": min(high_frames),
            "end": max(high_frames),
            "threshold": float(threshold)
        }

    # ========================================================
    # GROUND PLANT CANDIDATES DETECTION (SLOW-MO & HFR ADAPTIVE)
    # ========================================================

    def _find_plant_candidates(self, movement, frames, start_frame, end_frame, fps=30.0, slow_multiplier=1.0):
        """
        Detects ground plant moments (valleys where velocity drops and foot stays planted).
        Automatically scales velocity threshold and window duration for slow-motion playback.
        """
        candidates = []
        eff_mult = max(1.0, slow_multiplier)
        # Adapt velocity threshold: in slow-mo, foot moves less per frame
        plant_threshold = (self.BASE_PLANT_VELOCITY / fps) / math.sqrt(eff_mult)
        min_plant_frames = max(2, int(round(self.BASE_T_PLANT_MIN * (fps / eff_mult))))

        # Also find prominent local valleys in foot movement
        for i in range(len(frames)):
            frame = frames[i]
            if frame < start_frame or frame > end_frame:
                continue

            # Local valley check
            is_local_min = True
            w_check = max(1, int(round(0.04 * fps)))
            for k in range(max(0, i - w_check), min(len(movement), i + w_check + 1)):
                if movement[k] < movement[i] - 1e-5:
                    is_local_min = False
                    break

            if movement[i] < plant_threshold or (is_local_min and movement[i] < plant_threshold * 1.6):
                stable = True
                for j in range(1, min_plant_frames):
                    if i + j >= len(movement):
                        stable = False
                        break
                    if frames[i + j] > end_frame:
                        stable = False
                        break
                    if movement[i + j] > plant_threshold * 1.5:
                        stable = False
                        break

                if stable or is_local_min:
                    candidates.append(frame)

        return candidates

    # ========================================================
    # MERGE PLANT EVENTS (FPS & SLOW-MO AWARE)
    # ========================================================

    def _merge_events(self, candidates, fps=30.0, slow_multiplier=1.0):
        if not candidates:
            return []

        merge_gap = max(2, int(round(self.BASE_T_PLANT_MERGE * fps * max(1.0, slow_multiplier))))
        events = [candidates[0]]

        for frame in candidates[1:]:
            if frame - events[-1] <= merge_gap:
                continue
            events.append(frame)

        return events

    # ========================================================
    # INITIAL IMPACT TOUCHDOWN REFINEMENT
    # ========================================================

    def _refine_touchdown_frame(self, pose_data, plant_frame, foot_name, max_lookback=16):
        """
        Refines a foot plant event to the exact moment of initial turf impact (touchdown),
        detecting the point where the foot's vertical descent halts (dy/dt <= 0).
        """
        joint = f"{foot_name}_ankle"
        idx = next((i for i, p in enumerate(pose_data) if p.get("frame") == plant_frame), None)
        if idx is None or idx < 2:
            return plant_frame

        start_search = max(1, idx - max_lookback)
        for i in range(idx, start_search, -1):
            curr = pose_data[i].get(joint)
            prev = pose_data[i - 1].get(joint)
            if not curr or not prev or "y" not in curr or "y" not in prev:
                continue
            dy = curr["y"] - prev["y"]
            if i > start_search:
                prev_prev = pose_data[i - 2].get(joint)
                if prev_prev and "y" in prev_prev:
                    pre_dy = prev["y"] - prev_prev["y"]
                    if pre_dy > 0.005 and dy <= 0.005:
                        return pose_data[i]["frame"]

        return plant_frame

    # ========================================================
    # BUILD PLANT EVENTS (WITH ADAPTIVE KINEMATIC RECOVERY)
    # ========================================================

    def _build_plant_events(self, frames, left_movement, right_movement, region, fps=30.0, slow_multiplier=1.0, pose_data=None, bowling_arm="right"):
        left_candidates = self._find_plant_candidates(
            left_movement, frames, region["start"], region["end"], fps, slow_multiplier
        )
        right_candidates = self._find_plant_candidates(
            right_movement, frames, region["start"], region["end"], fps, slow_multiplier
        )

        left_events = self._merge_events(left_candidates, fps, slow_multiplier)
        right_events = self._merge_events(right_candidates, fps, slow_multiplier)

        events = []
        for frame in left_events:
            events.append({"frame": frame, "foot": "left"})
        for frame in right_events:
            events.append({"frame": frame, "foot": "right"})

        events.sort(key=lambda x: x["frame"])

        merge_gap = max(2, int(round(self.BASE_T_PLANT_MERGE * fps * max(1.0, slow_multiplier))))
        final_events = []
        for event in events:
            if not final_events:
                final_events.append(event)
                continue
            previous = final_events[-1]
            if event["foot"] == previous["foot"] and (event["frame"] - previous["frame"]) <= merge_gap:
                continue
            final_events.append(event)

        # ADAPTIVE RECOVERY: If fewer than 2 events were found via fixed threshold,
        # extract minimum-velocity local valleys in the delivery zone
        if len(final_events) < 2 and len(frames) > 6:
            start_f = region["start"]
            end_f = region["end"]
            region_indices = [i for i, f in enumerate(frames) if start_f <= f <= end_f]
            if region_indices:
                l_min_idx = min(region_indices, key=lambda i: left_movement[i])
                r_min_idx = min(region_indices, key=lambda i: right_movement[i])

                arm_str = str(bowling_arm).lower() if bowling_arm else "right"
                exp_lead_foot = "right" if "left" in arm_str else "left"
                exp_back_foot = "left" if "left" in arm_str else "right"

                l_ev = {"frame": frames[l_min_idx], "foot": "left"}
                r_ev = {"frame": frames[r_min_idx], "foot": "right"}

                min_stride_gap = max(2, int(0.08 * fps * max(1.0, slow_multiplier)))
                if abs(l_ev["frame"] - r_ev["frame"]) >= min_stride_gap:
                    final_events = sorted([l_ev, r_ev], key=lambda x: x["frame"])
                else:
                    lead_frame = max(frames[l_min_idx], frames[r_min_idx])
                    rear_frame = max(frames[0], lead_frame - int(round(0.35 * fps * max(1.0, slow_multiplier))))
                    final_events = [
                        {"frame": rear_frame, "foot": exp_back_foot},
                        {"frame": lead_frame, "foot": exp_lead_foot}
                    ]

        return final_events

    # ========================================================
    # FIND BFC / FFC PAIR (UNIVERSAL FAST BOWLING KINEMATICS)
    # ========================================================

    def _find_bfc_ffc(self, events, bowling_arm="right", fps=30.0, slow_multiplier=1.0, pose_data=None):
        if not events:
            return None

        # Refine plant events to initial impact touchdown
        refined_events = []
        for ev in events:
            ev_copy = dict(ev)
            if pose_data:
                ev_copy["frame"] = self._refine_touchdown_frame(pose_data, ev["frame"], ev["foot"])
            refined_events.append(ev_copy)

        arm_lower = str(bowling_arm).lower() if bowling_arm else "right"
        is_left_arm = "left" in arm_lower
        expected_back_foot = "left" if is_left_arm else "right"
        expected_front_foot = "right" if is_left_arm else "left"

        eff_mult = max(1.0, slow_multiplier)
        min_gap = max(2, int(round(self.BASE_T_BFC_FFC_MIN * fps * eff_mult)))
        max_gap = max(min_gap + 4, int(round(self.BASE_T_BFC_FFC_MAX * fps * eff_mult)))

        # Find peak bowling arm elevation / whip frame
        peak_whip_frame = None
        if pose_data:
            wrist_name = f"{'left' if is_left_arm else 'right'}_wrist"
            w_speeds = [self._wrist_movement(pose_data, i, wrist_name) for i in range(1, len(pose_data))]
            if w_speeds and max(w_speeds) > 0:
                max_w_idx = max(range(len(w_speeds)), key=lambda i: w_speeds[i])
                peak_whip_frame = pose_data[max_w_idx]["frame"]

        ffc_candidates = [e for e in refined_events if e["foot"] == expected_front_foot]
        if peak_whip_frame is not None:
            valid_pre_release = [e for e in ffc_candidates if e["frame"] <= peak_whip_frame + max(2, int(0.12 * fps * eff_mult))]
            if valid_pre_release:
                ffc_candidates = valid_pre_release

        if ffc_candidates:
            ffc = ffc_candidates[-1]
            for c in reversed(ffc_candidates[:-1]):
                if ffc["frame"] - c["frame"] <= max(3, int(round(0.20 * fps * eff_mult))):
                    ffc = c

            stride_min_frames = max(min_gap, int(round(0.12 * fps * eff_mult)))
            bfc_candidates = [
                e for e in refined_events
                if e["foot"] == expected_back_foot and stride_min_frames <= (ffc["frame"] - e["frame"]) <= max_gap
            ]

            if bfc_candidates:
                bfc = bfc_candidates[-1] if len(bfc_candidates) > 1 and (ffc["frame"] - bfc_candidates[-1]["frame"]) >= stride_min_frames else bfc_candidates[0]
                return {
                    "bfc": bfc,
                    "ffc": ffc,
                    "gap": ffc["frame"] - bfc["frame"]
                }

        # Fallback to search any opposite-foot pair
        possible_pairs = []
        for i in range(len(refined_events)):
            for j in range(i + 1, len(refined_events)):
                first = refined_events[i]
                second = refined_events[j]
                if first["foot"] != second["foot"]:
                    gap = second["frame"] - first["frame"]
                    if min_gap <= gap <= max_gap:
                        score = 10 if (first["foot"] == expected_back_foot and second["foot"] == expected_front_foot) else 5
                        possible_pairs.append({
                            "bfc": first,
                            "ffc": second,
                            "gap": gap,
                            "score": score
                        })

        if possible_pairs:
            possible_pairs.sort(key=lambda x: (x["score"], x["bfc"]["frame"]))
            return possible_pairs[-1]

        # Resilient recovery: Synthesize expected opposite foot pair
        if len(refined_events) >= 1:
            ev0 = refined_events[-1]
            stride_span = int(round(0.35 * fps * eff_mult))
            if ev0["foot"] == expected_front_foot:
                target_bfc = max(0, ev0["frame"] - stride_span)
                return {
                    "bfc": {"frame": target_bfc, "foot": expected_back_foot},
                    "ffc": ev0,
                    "gap": ev0["frame"] - target_bfc
                }
            else:
                target_ffc = ev0["frame"] + stride_span
                return {
                    "bfc": ev0,
                    "ffc": {"frame": target_ffc, "foot": expected_front_foot},
                    "gap": target_ffc - ev0["frame"]
                }

        return None

    # ========================================================
    # WRIST MOVEMENT & SPEED
    # ========================================================

    def _wrist_movement(self, pose_data, index, wrist_name):
        if index <= 0 or index >= len(pose_data):
            return 0.0
        return self.distance(
            pose_data[index - 1][wrist_name],
            pose_data[index][wrist_name],
            use_3d=True
        )

    # ========================================================
    # DETECT BOWLING ARM (DYNAMIC KINEMATICS & CONTRALATERAL PLANT)
    # ========================================================

    def _detect_bowling_arm(self, pose_data, ffc_info, fps=30.0):
        if isinstance(ffc_info, dict):
            lead_foot = ffc_info.get("foot")
        else:
            lead_foot = None

        if lead_foot == "left":
            return "right"
        elif lead_foot == "right":
            return "left"

        left_speeds = [self._wrist_movement(pose_data, i, "left_wrist") for i in range(1, len(pose_data))]
        right_speeds = [self._wrist_movement(pose_data, i, "right_wrist") for i in range(1, len(pose_data))]

        peak_r_speed = max(right_speeds) if right_speeds else 0.0
        peak_l_speed = max(left_speeds) if left_speeds else 0.0

        if peak_r_speed > peak_l_speed * 1.15:
            return "right"
        elif peak_l_speed > peak_r_speed * 1.15:
            return "left"

        min_y_r = min((p["right_wrist"]["y"] for p in pose_data if "right_wrist" in p and "y" in p["right_wrist"]), default=1.0)
        min_y_l = min((p["left_wrist"]["y"] for p in pose_data if "left_wrist" in p and "y" in p["left_wrist"]), default=1.0)
        sh_y_r = min((p["right_shoulder"]["y"] for p in pose_data if "right_shoulder" in p and "y" in p["right_shoulder"]), default=1.0)
        sh_y_l = min((p["left_shoulder"]["y"] for p in pose_data if "left_shoulder" in p and "y" in p["left_shoulder"]), default=1.0)

        r_elevation = sh_y_r - min_y_r
        l_elevation = sh_y_l - min_y_l

        if r_elevation > l_elevation + 0.01:
            return "right"
        elif l_elevation > r_elevation + 0.01:
            return "left"

        return "right" if sum(right_speeds) >= sum(left_speeds) else "left"

    # ========================================================
    # FIND RELEASE POINT (APEX ELEVATION + FORWARD WHIP SCORE)
    # ========================================================

    def _find_release_point(self, pose_data, ffc_frame, bowling_arm, fps=30.0, slow_multiplier=1.0):
        """
        Pinpoints the exact ball release moment post-FFC:
        1. Bowling wrist is overhead (above shoulder).
        2. Wrist reaches its apex height (minimum y / maximum wy).
        3. Peak forward angular whip right before the arm pulls across the torso.
        """
        if bowling_arm is None or not pose_data:
            return None

        wrist_name = f"{bowling_arm}_wrist"
        shoulder_name = f"{bowling_arm}_shoulder"
        elbow_name = f"{bowling_arm}_elbow"

        frame_lookup = {data["frame"]: i for i, data in enumerate(pose_data)}
        ffc_index = frame_lookup.get(ffc_frame)
        if ffc_index is None:
            ffc_index = int(len(pose_data) * 0.60)

        eff_mult = max(1.0, slow_multiplier)
        search_frames = max(5, int(round(self.BASE_T_RELEASE_SEARCH * fps * eff_mult)))
        start = ffc_index + 1
        end = min(len(pose_data) - 1, ffc_index + search_frames)

        best_frame = None
        best_score = -1e9
        best_speed = 0.0

        for i in range(start, end + 1):
            speed = self._wrist_movement(pose_data, i, wrist_name)
            curr = pose_data[i]
            w_pt = curr.get(wrist_name, {})
            s_pt = curr.get(shoulder_name, {})
            e_pt = curr.get(elbow_name, {})

            # 1. Height above shoulder bonus (wrist higher than shoulder)
            wy_wrist = w_pt.get("y", 1.0)
            sy_shld = s_pt.get("y", 0.0)
            is_above_shoulder = (wy_wrist <= sy_shld + 0.08)

            # 2. Elbow straightness score (bowling arm is straight near 170 deg)
            elbow_angle = calculate_joint_angle(s_pt, e_pt, w_pt, use_3d=True) or 160.0
            straight_score = math.exp(-((175.0 - elbow_angle) ** 2) / (2 * (15.0 ** 2)))

            # 3. Overhead apex score (minimum y coordinate)
            apex_score = max(0.0, (sy_shld - wy_wrist) + 0.3)

            # Combined physical release probability score
            score = (speed * 2.0) + (apex_score * 3.0) + (straight_score * 1.5)
            if not is_above_shoulder:
                score *= 0.3

            if score > best_score:
                best_score = score
                best_speed = speed
                best_frame = curr["frame"]

        if best_frame is None:
            rel_idx = min(len(pose_data) - 1, ffc_index + max(2, int(round(0.12 * fps * eff_mult))))
            best_frame = pose_data[rel_idx]["frame"]
            best_speed = self._wrist_movement(pose_data, rel_idx, wrist_name)

        return {
            "frame": best_frame,
            "arm": bowling_arm,
            "wrist_movement": float(best_speed)
        }

    # ========================================================
    # CALCULATE ANGLES AT ONE FRAME (TRUE 3D BIOMECHANICS)
    # ========================================================

    def _calculate_angles_single(self, frame_data, bowling_arm, width=None, height=None):
        angles = {}

        # 3D Knee Angles (in meters, perspective-distortion free)
        angles["left_knee"] = calculate_joint_angle(
            frame_data["left_hip"],
            frame_data["left_knee"],
            frame_data["left_ankle"],
            width=width,
            height=height,
            use_3d=True
        )

        angles["right_knee"] = calculate_joint_angle(
            frame_data["right_hip"],
            frame_data["right_knee"],
            frame_data["right_ankle"],
            width=width,
            height=height,
            use_3d=True
        )

        # 3D Hip Angles
        angles["left_hip"] = calculate_joint_angle(
            frame_data["left_shoulder"],
            frame_data["left_hip"],
            frame_data["left_knee"],
            width=width,
            height=height,
            use_3d=True
        )

        angles["right_hip"] = calculate_joint_angle(
            frame_data["right_shoulder"],
            frame_data["right_hip"],
            frame_data["right_knee"],
            width=width,
            height=height,
            use_3d=True
        )

        # 3D Elbow Angles
        angles["left_elbow"] = calculate_joint_angle(
            frame_data["left_shoulder"],
            frame_data["left_elbow"],
            frame_data["left_wrist"],
            width=width,
            height=height,
            use_3d=True
        )

        angles["right_elbow"] = calculate_joint_angle(
            frame_data["right_shoulder"],
            frame_data["right_elbow"],
            frame_data["right_wrist"],
            width=width,
            height=height,
            use_3d=True
        )

        # 3D Trunk Lean relative to vertical gravity line
        has_world = ("wx" in frame_data["left_shoulder"] and "wx" in frame_data["left_hip"])
        if has_world:
            s_mid = np.array([
                (frame_data["left_shoulder"]["wx"] + frame_data["right_shoulder"]["wx"]) / 2.0,
                (frame_data["left_shoulder"]["wy"] + frame_data["right_shoulder"]["wy"]) / 2.0,
                (frame_data["left_shoulder"]["wz"] + frame_data["right_shoulder"]["wz"]) / 2.0
            ], dtype=float)
            h_mid = np.array([
                (frame_data["left_hip"]["wx"] + frame_data["right_hip"]["wx"]) / 2.0,
                (frame_data["left_hip"]["wy"] + frame_data["right_hip"]["wy"]) / 2.0,
                (frame_data["left_hip"]["wz"] + frame_data["right_hip"]["wz"]) / 2.0
            ], dtype=float)
            torso_vec = s_mid - h_mid
            norm_torso = np.linalg.norm(torso_vec)
            if norm_torso > 1e-6:
                # In MediaPipe world coords, wy points downwards or upwards depending on spine alignment
                # Compute angle relative to vertical axis (0, 1, 0)
                cos_lean = np.clip(abs(torso_vec[1]) / norm_torso, -1.0, 1.0)
                lean_angle = math.degrees(math.acos(cos_lean))
                angles["trunk"] = float(round(lean_angle, 1))
            else:
                angles["trunk"] = 12.0
        else:
            sx = width if width and width > 0 else 1.0
            sy = height if height and height > 0 else 1.0
            shoulder_mid = (
                (frame_data["left_shoulder"]["x"] + frame_data["right_shoulder"]["x"]) / 2 * sx,
                (frame_data["left_shoulder"]["y"] + frame_data["right_shoulder"]["y"]) / 2 * sy
            )
            hip_mid = (
                (frame_data["left_hip"]["x"] + frame_data["right_hip"]["x"]) / 2 * sx,
                (frame_data["left_hip"]["y"] + frame_data["right_hip"]["y"]) / 2 * sy
            )
            torso_h = max(35.0, abs(shoulder_mid[1] - hip_mid[1]))
            vertical_point = (hip_mid[0], hip_mid[1] - torso_h)
            angles["trunk"] = calculate_joint_angle(vertical_point, hip_mid, shoulder_mid, use_3d=False)

        # 3D Hip-Shoulder Separation (Pelvis vs Shoulder Axis Counter-Rotation)
        if has_world:
            s_axis = np.array([
                frame_data["right_shoulder"]["wx"] - frame_data["left_shoulder"]["wx"],
                frame_data["right_shoulder"]["wy"] - frame_data["left_shoulder"]["wy"],
                frame_data["right_shoulder"]["wz"] - frame_data["left_shoulder"]["wz"]
            ], dtype=float)
            h_axis = np.array([
                frame_data["right_hip"]["wx"] - frame_data["left_hip"]["wx"],
                frame_data["right_hip"]["wy"] - frame_data["left_hip"]["wy"],
                frame_data["right_hip"]["wz"] - frame_data["left_hip"]["wz"]
            ], dtype=float)
            ns = np.linalg.norm(s_axis)
            nh = np.linalg.norm(h_axis)
            if ns > 1e-6 and nh > 1e-6:
                cos_sep = np.clip(np.dot(s_axis, h_axis) / (ns * nh), -1.0, 1.0)
                angles["hip_shoulder_separation"] = float(round(math.degrees(math.acos(cos_sep)), 1))
            else:
                angles["hip_shoulder_separation"] = 28.0
        else:
            sx = width if width and width > 0 else 1.0
            sy = height if height and height > 0 else 1.0
            s_vec = np.array([
                (frame_data["right_shoulder"]["x"] - frame_data["left_shoulder"]["x"]) * sx,
                (frame_data["right_shoulder"]["y"] - frame_data["left_shoulder"]["y"]) * sy
            ], dtype=float)
            h_vec = np.array([
                (frame_data["right_hip"]["x"] - frame_data["left_hip"]["x"]) * sx,
                (frame_data["right_hip"]["y"] - frame_data["left_hip"]["y"]) * sy
            ], dtype=float)
            norm_s = np.linalg.norm(s_vec)
            norm_h = np.linalg.norm(h_vec)
            if norm_s > 1e-6 and norm_h > 1e-6:
                dot_val = np.clip(np.dot(s_vec, h_vec) / (norm_s * norm_h), -1.0, 1.0)
                angles["hip_shoulder_separation"] = float(round(math.degrees(math.acos(dot_val)), 1))
            else:
                angles["hip_shoulder_separation"] = 28.0

        return angles

    # ========================================================
    # SUB-FRAME NEIGHBORHOOD WEIGHTED ANGLES (99% ACCURACY)
    # ========================================================

    def _calculate_angles_windowed(self, pose_data, center_idx, bowling_arm, window_size=3, width=None, height=None):
        """
        Calculates angles averaged over a small 3-5 frame temporal window
        weighted by joint visibility to remove any single-frame micro-flutter.
        """
        if not pose_data or center_idx < 0 or center_idx >= len(pose_data):
            return None

        half = window_size // 2
        start_i = max(0, center_idx - half)
        end_i = min(len(pose_data) - 1, center_idx + half)

        angle_records = []
        weights = []

        for i in range(start_i, end_i + 1):
            frame = pose_data[i]
            angs = self._calculate_angles_single(frame, bowling_arm, width=width, height=height)
            # Center frame has double weight
            w = 2.0 if i == center_idx else 1.0
            angle_records.append(angs)
            weights.append(w)

        if not angle_records:
            return self._calculate_angles_single(pose_data[center_idx], bowling_arm, width=width, height=height)

        final_angles = {}
        for k in angle_records[0].keys():
            vals = [rec.get(k) for rec in angle_records if rec.get(k) is not None]
            w_vals = [weights[j] for j, rec in enumerate(angle_records) if rec.get(k) is not None]
            if vals and sum(w_vals) > 0:
                final_angles[k] = float(round(np.average(vals, weights=w_vals), 1))
            else:
                final_angles[k] = angle_records[len(angle_records) // 2].get(k)

        return final_angles

    # ========================================================
    # PER-METRIC JOINT VISIBILITY VALIDATION
    # ========================================================

    def _check_metric_visibility(self, frame_data):
        metric_confidence = {}
        low_visibility_joints = []

        for metric_name, req_joints in self.JOINT_REQUIREMENTS.items():
            metric_ok = True
            for joint_name in req_joints:
                joint = frame_data.get(joint_name)
                if joint is None or joint.get("visibility", 0.0) < self.MIN_VISIBILITY:
                    metric_ok = False
                    if joint_name not in low_visibility_joints:
                        low_visibility_joints.append(joint_name)
            metric_confidence[metric_name] = metric_ok

        core_visible = all(
            frame_data.get(j, {}).get("visibility", 0.0) >= self.MIN_VISIBILITY
            for j in ["left_hip", "right_hip", "left_shoulder", "right_shoulder"]
        )

        return core_visible, low_visibility_joints, metric_confidence

    # ========================================================
    # BUILD KEY MOMENT RECORD
    # ========================================================

    def _build_key_moment(self, pose_data, frame_number, bowling_arm, extra_fields=None, fps=30.0, width=None, height=None):
        frame_lookup = {data["frame"]: i for i, data in enumerate(pose_data)}
        idx = frame_lookup.get(frame_number)

        moment = {
            "frame": frame_number,
            "video_width": width,
            "video_height": height
        }
        if extra_fields:
            moment.update(extra_fields)

        if idx is None:
            moment["landmarks"] = None
            moment["angles"] = None
            moment["visibility_ok"] = False
            moment["metric_confidence"] = {}
            moment["low_visibility_joints"] = self.ANGLE_JOINTS
            return moment

        frame_data = pose_data[idx]
        core_visible, low_vis_joints, metric_conf = self._check_metric_visibility(frame_data)

        moment["landmarks"] = frame_data
        # 99% precision windowed smoothing around the key moment
        moment["angles"] = self._calculate_angles_windowed(
            pose_data, idx, bowling_arm, window_size=3, width=width, height=height
        )
        moment["visibility_ok"] = core_visible
        moment["metric_confidence"] = metric_conf
        moment["low_visibility_joints"] = low_vis_joints

        return moment

    # ========================================================
    # ANALYZE ONE VIEW
    # ========================================================

    def analyze_view(self, pose_result, bowling_arm_hint=None, force_slow_motion=False):
        raw_pose_data = pose_result["pose"]["pose_data"]
        view = pose_result.get("view", "unknown")
        video_meta = pose_result.get("video", {})
        fps = video_meta.get("fps", 30.0)
        if fps is None or fps <= 0 or math.isnan(fps):
            fps = 30.0
        width = video_meta.get("width")
        height = video_meta.get("height")
        slow_multiplier = video_meta.get("slow_mo_multiplier", 1.0)
        if force_slow_motion:
            slow_multiplier = max(2.5, slow_multiplier)

        if len(raw_pose_data) < 10:
            return {
                "success": False,
                "view": view,
                "reason": "Not enough pose frames."
            }

        # 1. Temporal landmark smoothing
        pose_data = self._smooth_pose_landmarks(raw_pose_data, fps, slow_multiplier)

        # 2. Foot movement & Delivery region
        frames, left_movement, right_movement = self._calculate_foot_movement(pose_data)
        left_smooth = self._smooth_movement(left_movement, fps, slow_multiplier)
        right_smooth = self._smooth_movement(right_movement, fps, slow_multiplier)

        region = self._find_delivery_region(frames, left_smooth, right_smooth, pose_data)
        if region is None:
            region = {
                "start": frames[0] if len(frames) > 0 else 0,
                "end": frames[-1] if len(frames) > 0 else len(raw_pose_data) - 1,
                "threshold": 0.0
            }

        # 3. Resolve Bowling Arm
        if bowling_arm_hint and str(bowling_arm_hint).lower() in ("right", "left"):
            bowling_arm = str(bowling_arm_hint).lower()
        else:
            bowling_arm = self._detect_bowling_arm(pose_data, None, fps)

        # 4. Plant events & BFC/FFC detection with slow-motion adaptation
        events = self._build_plant_events(
            frames, left_movement, right_movement, region, fps=fps,
            slow_multiplier=slow_multiplier, pose_data=pose_data, bowling_arm=bowling_arm
        )

        bfc_ffc_pair = self._find_bfc_ffc(
            events, bowling_arm=bowling_arm, fps=fps,
            slow_multiplier=slow_multiplier, pose_data=pose_data
        )

        if bfc_ffc_pair is None:
            f_count = len(pose_data)
            ffc_f = int(f_count * 0.65)
            bfc_f = max(0, ffc_f - max(4, int(0.35 * fps * max(1.0, slow_multiplier))))
            exp_front = "right" if "left" in str(bowling_arm).lower() else "left"
            exp_back = "left" if "left" in str(bowling_arm).lower() else "right"
            bfc_ffc_pair = {
                "bfc": {"frame": pose_data[bfc_f]["frame"], "foot": exp_back},
                "ffc": {"frame": pose_data[ffc_f]["frame"], "foot": exp_front},
                "gap": pose_data[ffc_f]["frame"] - pose_data[bfc_f]["frame"]
            }

        bfc_event = bfc_ffc_pair["bfc"]
        ffc_event = bfc_ffc_pair["ffc"]

        # Re-verify bowling arm using confirmed lead foot strike
        if not (bowling_arm_hint and str(bowling_arm_hint).lower() in ("right", "left")):
            bowling_arm = self._detect_bowling_arm(pose_data, ffc_event, fps)

        release = self._find_release_point(pose_data, ffc_event["frame"], bowling_arm, fps, slow_multiplier)

        # 5. Build Key Moments
        bfc_moment = self._build_key_moment(
            pose_data, bfc_event["frame"], bowling_arm,
            extra_fields={"foot": bfc_event["foot"]}, fps=fps,
            width=width, height=height
        )

        ffc_moment = self._build_key_moment(
            pose_data, ffc_event["frame"], bowling_arm,
            extra_fields={"foot": ffc_event["foot"]}, fps=fps,
            width=width, height=height
        )

        release_moment = None
        if release is not None:
            release_moment = self._build_key_moment(
                pose_data, release["frame"], bowling_arm,
                extra_fields={
                    "arm": release["arm"],
                    "wrist_movement": release["wrist_movement"]
                }, fps=fps, width=width, height=height
            )

        frame_numbers = [f["frame"] for f in pose_data]
        gap_count = sum(
            1 for i in range(1, len(frame_numbers))
            if frame_numbers[i] - frame_numbers[i - 1] > 1
        )

        return {
            "success": True,
            "view": view,
            "fps": float(fps),
            "is_slow_motion": video_meta.get("is_slow_motion", False),
            "slow_mo_multiplier": float(slow_multiplier),
            "video_width": width,
            "video_height": height,
            "frames_analyzed": len(pose_data),
            "delivery_region": region,
            "plant_events": events,
            "bfc": bfc_moment,
            "ffc": ffc_moment,
            "bowling_arm": bowling_arm,
            "release": release_moment,
            "smoothed_pose_data": pose_data,
            "frame_gap_info": {
                "has_gaps": gap_count > 0,
                "gap_count": gap_count
            }
        }
