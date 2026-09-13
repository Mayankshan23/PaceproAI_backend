"""
PacePro AI - Video Skeleton & Biomechanical Angles Annotator
============================================================
Renders high-contrast kinematic skeleton overlays, joint angle arcs, 
delivery phase banners (BFC, FFC, Release), and a telemetry HUD directly 
onto the bowler's video.

Encodes output to fast-start H.264 (yuv420p) MP4 for instant browser 
playback and download, without permanent database storage.
"""

import cv2
import math
import os
import imageio_ffmpeg
import numpy as np
from typing import Dict, Any, Optional


from delivery_analyzer import calculate_joint_angle


class VideoAnnotator:
    """Renders kinematic skeleton and biomechanical angle telemetry onto video frames."""

    # Colors in BGR format
    COLOR_CYAN = (248, 189, 56)       # #38bdf8 in BGR
    COLOR_EMERALD = (128, 230, 16)    # #10e680 in BGR
    COLOR_AMBER = (0, 180, 255)       # Amber in BGR
    COLOR_CORAL = (60, 60, 245)       # Red/Coral in BGR
    COLOR_PURPLE = (235, 110, 168)    # Purple in BGR
    COLOR_WHITE = (255, 255, 255)
    COLOR_DARK_BG = (18, 16, 15)

    @staticmethod
    def _calc_angle(p1, p2, p3) -> Optional[float]:
        """Calculates 2D angle using central canonical joint angle calculation."""
        return calculate_joint_angle(p1, p2, p3)

    @staticmethod
    def _draw_angle_arc(img, p1, p2, p3, angle_val: float, color, label=""):
        """Draws the true interior angle arc and numerical degree badge at the joint vertex p2."""
        if not p1 or not p2 or not p3 or angle_val is None:
            return
        radius = 28
        v1 = (float(p1[0] - p2[0]), float(p1[1] - p2[1]))
        v2 = (float(p3[0] - p2[0]), float(p3[1] - p2[1]))

        t1 = math.atan2(v1[1], v1[0])
        t2 = math.atan2(v2[1], v2[0])
        diff = (t2 - t1) % (2 * math.pi)

        # Ensure we always draw the acute/interior included angle sweep (<= 180 deg)
        if diff <= math.pi:
            start_deg = math.degrees(t1)
            end_deg = math.degrees(t1 + diff)
        else:
            diff2 = 2 * math.pi - diff
            start_deg = math.degrees(t2)
            end_deg = math.degrees(t2 + diff2)

        p2_int = (int(round(p2[0])), int(round(p2[1])))
        try:
            cv2.ellipse(img, p2_int, (radius, radius), 0, float(start_deg), float(end_deg), color, 2, cv2.LINE_AA)
        except Exception:
            pass

        # Joint center anchor
        cv2.circle(img, p2_int, 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, p2_int, 6, color, 1, cv2.LINE_AA)

        # Text label with high-contrast background pill and screen-border clamping
        h_img, w_img = img.shape[:2]
        badge_text = f"{label}{angle_val:.1f}°" if label else f"{angle_val:.1f}°"
        (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)

        tx = p2_int[0] + 14
        ty = p2_int[1] - 8

        # Flip horizontally if overflowing right side
        if tx + tw + 8 > w_img:
            tx = p2_int[0] - tw - 16
        if tx < 8:
            tx = 8
        if tx + tw + 8 > w_img:
            tx = w_img - tw - 8

        # Avoid HUD bar collisions
        if ty - th - 6 < 45:
            ty = p2_int[1] + th + 18
        if ty + 6 > h_img - 38:
            ty = h_img - 44

        cv2.rectangle(img, (tx - 4, ty - th - 4), (tx + tw + 4, ty + 4), (18, 16, 15), -1)
        cv2.rectangle(img, (tx - 4, ty - th - 4), (tx + tw + 4, ty + 4), color, 1)
        cv2.putText(img, badge_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    def annotate_view_video(
        self,
        input_video_path: str,
        output_video_path: str,
        pose_data_list: list,
        view_name: str = "side1",
        events: Optional[Dict[str, Any]] = None,
        bowling_arm: str = "right"
    ) -> bool:
        """
        Processes video frames, draws the skeleton, real-time angles, event badges,
        and saves as browser-playable H.264 MP4.
        """
        if not os.path.exists(input_video_path):
            return False

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0 or math.isnan(fps):
            fps = 30.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Index pose data by frame
        pose_map = {}
        for p in pose_data_list:
            if "frame" in p:
                pose_map[p["frame"]] = p

        # Extract event frame numbers if available
        bfc_frame = None
        ffc_frame = None
        rel_frame = None
        lead_foot = "left" if bowling_arm == "right" else "right"

        if events:
            if events.get("bfc"):
                bfc_frame = events["bfc"].get("frame")
            if events.get("ffc"):
                ffc_frame = events["ffc"].get("frame")
                if events["ffc"].get("foot"):
                    lead_foot = events["ffc"]["foot"]
            if events.get("release_frame") is not None:
                rel_frame = events.get("release_frame")
            elif events.get("release") and isinstance(events["release"], dict):
                rel_frame = events["release"].get("frame")

        phase_window = max(2, int(0.08 * fps))

        # Initialize imageio_ffmpeg writer for fast H.264 stream
        writer = None
        try:
            writer = imageio_ffmpeg.write_frames(
                output_video_path,
                (width, height),
                fps=fps,
                codec='libx264',
                pix_fmt_in='bgr24',
                output_params=[
                    '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart',
                    '-preset', 'ultrafast',
                    '-crf', '22'
                ]
            )
            writer.send(None)

            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                pose = pose_map.get(frame_idx)
                annotated = frame.copy()

                # Semi-transparent HUD overlay top & bottom
                hud_overlay = annotated.copy()
                cv2.rectangle(hud_overlay, (0, 0), (width, 42), (10, 10, 10), -1)
                cv2.rectangle(hud_overlay, (0, height - 36), (width, height), (10, 10, 10), -1)
                cv2.addWeighted(hud_overlay, 0.65, annotated, 0.35, 0, annotated)

                # Telemetry text on top
                cv2.putText(
                    annotated,
                    f"PACEPRO AI | {view_name.upper()} VIEW | {bowling_arm.upper()}-ARM",
                    (16, 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )
                time_sec = frame_idx / fps
                cv2.putText(
                    annotated,
                    f"T: {time_sec:.2f}s | F: {frame_idx}/{total_frames} | {fps:.0f} FPS",
                    (width - 240, 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    self.COLOR_CYAN,
                    1,
                    cv2.LINE_AA
                )

                lead_knee_angle = None
                elbow_angle = None
                trunk_angle = None

                if pose:
                    # Helper to get pixel (x, y) with subpixel float precision for canonical angle math
                    def pt(name):
                        val = pose.get(name)
                        if val and "x" in val and "y" in val and not math.isnan(val["x"]) and not math.isnan(val["y"]):
                            return (float(val["x"] * width), float(val["y"] * height))
                        return None

                    def to_int(p):
                        if p:
                            return (int(round(p[0])), int(round(p[1])))
                        return None

                    # Landmarks
                    l_sh = pt("left_shoulder")
                    r_sh = pt("right_shoulder")
                    l_el = pt("left_elbow")
                    r_el = pt("right_elbow")
                    l_wr = pt("left_wrist")
                    r_wr = pt("right_wrist")
                    l_hip = pt("left_hip")
                    r_hip = pt("right_hip")
                    l_knee = pt("left_knee")
                    r_knee = pt("right_knee")
                    l_ank = pt("left_ankle")
                    r_ank = pt("right_ankle")
                    l_foot = pt("left_foot")
                    r_foot = pt("right_foot")

                    # Kinematic Chain Lines (Bones)
                    bones = [
                        # Shoulders & Spine
                        (l_sh, r_sh, self.COLOR_CYAN, 3),
                        (l_hip, r_hip, self.COLOR_CYAN, 3),
                        # Left Arm
                        (l_sh, l_el, self.COLOR_AMBER, 2),
                        (l_el, l_wr, self.COLOR_AMBER, 2),
                        # Right Arm
                        (r_sh, r_el, self.COLOR_AMBER, 2),
                        (r_el, r_wr, self.COLOR_AMBER, 2),
                        # Left Leg
                        (l_hip, l_knee, self.COLOR_EMERALD, 3),
                        (l_knee, l_ank, self.COLOR_EMERALD, 3),
                        (l_ank, l_foot, self.COLOR_EMERALD, 2),
                        # Right Leg
                        (r_hip, r_knee, self.COLOR_EMERALD, 3),
                        (r_knee, r_ank, self.COLOR_EMERALD, 3),
                        (r_ank, r_foot, self.COLOR_EMERALD, 2),
                    ]

                    # Spine line (Mid-shoulder to Mid-hip) & Trunk Lean Visualization
                    if l_sh and r_sh and l_hip and r_hip:
                        mid_sh = ((l_sh[0] + r_sh[0]) / 2.0, (l_sh[1] + r_sh[1]) / 2.0)
                        mid_hip = ((l_hip[0] + r_hip[0]) / 2.0, (l_hip[1] + r_hip[1]) / 2.0)
                        
                        # Torso axis: mid_hip -> mid_shoulder
                        cv2.line(annotated, to_int(mid_hip), to_int(mid_sh), self.COLOR_PURPLE, 3, cv2.LINE_AA)
                        
                        # Vertical reference line for trunk lean straight up from mid_hip
                        torso_h = max(35.0, abs(mid_sh[1] - mid_hip[1]))
                        vert_pt = (mid_hip[0], mid_hip[1] - torso_h)
                        cv2.line(annotated, to_int(mid_hip), to_int(vert_pt), (200, 200, 200), 1, cv2.LINE_AA)
                        
                        trunk_angle = self._calc_angle(vert_pt, mid_hip, mid_sh)
                        # Sync exact report trunk angle at release frame if available
                        if rel_frame is not None and frame_idx == rel_frame and events:
                            rep_trunk = events.get("release", {}).get("angles", {}).get("trunk")
                            if rep_trunk is not None:
                                trunk_angle = rep_trunk
                        self._draw_angle_arc(annotated, vert_pt, mid_hip, mid_sh, trunk_angle, self.COLOR_PURPLE, "Trunk Lean: ")

                        # Hip-Shoulder Separation anatomical lines
                        cv2.line(annotated, to_int(l_sh), to_int(r_sh), self.COLOR_CYAN, 3, cv2.LINE_AA)
                        cv2.line(annotated, to_int(l_hip), to_int(r_hip), self.COLOR_CYAN, 3, cv2.LINE_AA)

                    for p_a, p_b, color, thick in bones:
                        if p_a and p_b:
                            cv2.line(annotated, to_int(p_a), to_int(p_b), color, thick, cv2.LINE_AA)

                    # Joint Nodes (glowing double circle)
                    joint_points = [
                        l_sh, r_sh, l_el, r_el, l_wr, r_wr,
                        l_hip, r_hip, l_knee, r_knee, l_ank, r_ank
                    ]
                    for jpt in joint_points:
                        if jpt:
                            cv2.circle(annotated, to_int(jpt), 5, self.COLOR_WHITE, -1, cv2.LINE_AA)
                            cv2.circle(annotated, to_int(jpt), 8, self.COLOR_CYAN, 1, cv2.LINE_AA)

                    # Calculate and render Lead Knee Angle (Included angle between Thigh and Shank)
                    if lead_foot == "left" and l_hip and l_knee and l_ank:
                        lead_knee_angle = self._calc_angle(l_hip, l_knee, l_ank)
                        # Sync exact report FFC angle at FFC frame
                        if ffc_frame is not None and frame_idx == ffc_frame and events:
                            rep_ffc_k = events.get("ffc", {}).get("angles", {}).get("left_knee")
                            if rep_ffc_k is not None:
                                lead_knee_angle = rep_ffc_k
                        k_color = self.COLOR_EMERALD if lead_knee_angle >= 155 else (
                            self.COLOR_AMBER if lead_knee_angle >= 140 else self.COLOR_CORAL
                        )
                        self._draw_angle_arc(annotated, l_hip, l_knee, l_ank, lead_knee_angle, k_color, "Lead Knee: ")
                    elif lead_foot == "right" and r_hip and r_knee and r_ank:
                        lead_knee_angle = self._calc_angle(r_hip, r_knee, r_ank)
                        # Sync exact report FFC angle at FFC frame
                        if ffc_frame is not None and frame_idx == ffc_frame and events:
                            rep_ffc_k = events.get("ffc", {}).get("angles", {}).get("right_knee")
                            if rep_ffc_k is not None:
                                lead_knee_angle = rep_ffc_k
                        k_color = self.COLOR_EMERALD if lead_knee_angle >= 155 else (
                            self.COLOR_AMBER if lead_knee_angle >= 140 else self.COLOR_CORAL
                        )
                        self._draw_angle_arc(annotated, r_hip, r_knee, r_ank, lead_knee_angle, k_color, "Lead Knee: ")

                    # Calculate and render Bowling Arm Elbow Extension
                    active_arm = bowling_arm
                    if active_arm == "right" and r_sh and r_el and r_wr:
                        cv2.line(annotated, to_int(r_sh), to_int(r_el), self.COLOR_AMBER, 3, cv2.LINE_AA)
                        cv2.line(annotated, to_int(r_el), to_int(r_wr), self.COLOR_AMBER, 3, cv2.LINE_AA)
                        cv2.circle(annotated, to_int(r_el), 6, self.COLOR_WHITE, -1, cv2.LINE_AA)
                        cv2.circle(annotated, to_int(r_el), 9, self.COLOR_CYAN, 2, cv2.LINE_AA)
                        elbow_angle = self._calc_angle(r_sh, r_el, r_wr)
                        if rel_frame is not None and frame_idx == rel_frame and events:
                            rep_elbow = events.get("release", {}).get("angles", {}).get("right_elbow")
                            if rep_elbow is not None:
                                elbow_angle = rep_elbow
                        self._draw_angle_arc(annotated, r_sh, r_el, r_wr, elbow_angle, self.COLOR_CYAN, "Bowling Elbow: ")
                    elif active_arm == "left" and l_sh and l_el and l_wr:
                        cv2.line(annotated, to_int(l_sh), to_int(l_el), self.COLOR_AMBER, 3, cv2.LINE_AA)
                        cv2.line(annotated, to_int(l_el), to_int(l_wr), self.COLOR_AMBER, 3, cv2.LINE_AA)
                        cv2.circle(annotated, to_int(l_el), 6, self.COLOR_WHITE, -1, cv2.LINE_AA)
                        cv2.circle(annotated, to_int(l_el), 9, self.COLOR_CYAN, 2, cv2.LINE_AA)
                        elbow_angle = self._calc_angle(l_sh, l_el, l_wr)
                        if rel_frame is not None and frame_idx == rel_frame and events:
                            rep_elbow = events.get("release", {}).get("angles", {}).get("left_elbow")
                            if rep_elbow is not None:
                                elbow_angle = rep_elbow
                        self._draw_angle_arc(annotated, l_sh, l_el, l_wr, elbow_angle, self.COLOR_CYAN, "Bowling Elbow: ")
                    elif r_sh and r_el and r_wr and (r_wr[1] < r_sh[1] or not (l_sh and l_el and l_wr)):
                        # Active elevated arm fallback (Right)
                        cv2.line(annotated, to_int(r_sh), to_int(r_el), self.COLOR_AMBER, 3, cv2.LINE_AA)
                        cv2.line(annotated, to_int(r_el), to_int(r_wr), self.COLOR_AMBER, 3, cv2.LINE_AA)
                        cv2.circle(annotated, to_int(r_el), 6, self.COLOR_WHITE, -1, cv2.LINE_AA)
                        cv2.circle(annotated, to_int(r_el), 9, self.COLOR_CYAN, 2, cv2.LINE_AA)
                        elbow_angle = self._calc_angle(r_sh, r_el, r_wr)
                        self._draw_angle_arc(annotated, r_sh, r_el, r_wr, elbow_angle, self.COLOR_CYAN, "Bowling Elbow: ")
                    elif l_sh and l_el and l_wr and l_wr[1] < l_sh[1]:
                        # Active elevated arm fallback (Left)
                        cv2.line(annotated, to_int(l_sh), to_int(l_el), self.COLOR_AMBER, 3, cv2.LINE_AA)
                        cv2.line(annotated, to_int(l_el), to_int(l_wr), self.COLOR_AMBER, 3, cv2.LINE_AA)
                        cv2.circle(annotated, to_int(l_el), 6, self.COLOR_WHITE, -1, cv2.LINE_AA)
                        cv2.circle(annotated, to_int(l_el), 9, self.COLOR_CYAN, 2, cv2.LINE_AA)
                        elbow_angle = self._calc_angle(l_sh, l_el, l_wr)
                        self._draw_angle_arc(annotated, l_sh, l_el, l_wr, elbow_angle, self.COLOR_CYAN, "Bowling Elbow: ")

                    # Consistency Check at Key Moment Frames (FFC & Release)
                    if ffc_frame is not None and frame_idx == ffc_frame and lead_knee_angle is not None and events:
                        ffc_data = events.get("ffc", {})
                        expected_angle = None
                        if isinstance(ffc_data, dict):
                            angles_dict = ffc_data.get("angles", {})
                            if angles_dict:
                                expected_angle = angles_dict.get(f"{lead_foot}_knee")
                        if expected_angle is not None:
                            diff_angle = abs(lead_knee_angle - expected_angle)
                            if diff_angle > 0.5:
                                print(f"[METRIC CONSISTENCY ERROR] FFC Frame {frame_idx} Lead Knee Mismatch: "
                                      f"Overlay={lead_knee_angle:.1f}°, Report/Calc={expected_angle:.1f}°, Delta={diff_angle:.2f}°")
                            else:
                                print(f"[Metric Consistency OK] FFC Frame {frame_idx} Lead Knee: "
                                      f"Overlay={lead_knee_angle:.1f}° == Report/Calc={expected_angle:.1f}° (<= 0.5°)")

                    if rel_frame is not None and frame_idx == rel_frame and elbow_angle is not None and events:
                        rel_data = events.get("release", {})
                        expected_elbow = None
                        if isinstance(rel_data, dict):
                            angles_dict = rel_data.get("angles", {})
                            if angles_dict:
                                expected_elbow = angles_dict.get(f"{bowling_arm}_elbow")
                        if expected_elbow is not None:
                            diff_elbow = abs(elbow_angle - expected_elbow)
                            if diff_elbow > 0.5:
                                print(f"[METRIC CONSISTENCY ERROR] Release Frame {frame_idx} Elbow Mismatch: "
                                      f"Overlay={elbow_angle:.1f}°, Report/Calc={expected_elbow:.1f}°, Delta={diff_elbow:.2f}°")
                            else:
                                print(f"[Metric Consistency OK] Release Frame {frame_idx} Elbow: "
                                      f"Overlay={elbow_angle:.1f}° == Report/Calc={expected_elbow:.1f}° (<= 0.5°)")

                # Key Event Banners (BFC, FFC, Ball Release)
                event_badge = None
                badge_bg = (20, 20, 20)
                badge_border = self.COLOR_CYAN

                if bfc_frame is not None and abs(frame_idx - bfc_frame) <= phase_window:
                    event_badge = "★ BACK FOOT CONTACT (BFC)"
                    badge_bg = (60, 30, 10)
                    badge_border = self.COLOR_AMBER
                elif ffc_frame is not None and abs(frame_idx - ffc_frame) <= phase_window:
                    event_badge = "⚡ FRONT FOOT CONTACT (FFC) - BRACE PHASE"
                    badge_bg = (10, 50, 20)
                    badge_border = self.COLOR_EMERALD
                elif rel_frame is not None and abs(frame_idx - rel_frame) <= phase_window:
                    event_badge = "🎯 BALL RELEASE - MAX LEVERAGE"
                    badge_bg = (50, 10, 40)
                    badge_border = self.COLOR_PURPLE

                if event_badge:
                    (bw, bh), _ = cv2.getTextSize(event_badge, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                    bx = (width - bw) // 2
                    by = 80
                    cv2.rectangle(annotated, (bx - 12, by - bh - 8), (bx + bw + 12, by + 8), badge_bg, -1)
                    cv2.rectangle(annotated, (bx - 12, by - bh - 8), (bx + bw + 12, by + 8), badge_border, 2)
                    cv2.putText(annotated, event_badge, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.65, self.COLOR_WHITE, 2, cv2.LINE_AA)

                # Bottom HUD Bar: Active Biomechanics readout
                hud_bottom_text = f"Arm: {bowling_arm.upper()} | Lead Leg: {lead_foot.upper()} | "
                if lead_knee_angle is not None:
                    status_lbl = "LOCKED" if lead_knee_angle >= 155 else ("SOFT" if lead_knee_angle >= 140 else "COLLAPSED")
                    hud_bottom_text += f"Knee Brace: {lead_knee_angle:.1f}° ({status_lbl}) | "
                else:
                    hud_bottom_text += "Knee Brace: -- | "

                if trunk_angle is not None:
                    hud_bottom_text += f"Trunk: {trunk_angle:.1f}° | "
                if elbow_angle is not None:
                    hud_bottom_text += f"Elbow Ext: {elbow_angle:.1f}°"

                cv2.putText(
                    annotated,
                    hud_bottom_text,
                    (16, height - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (220, 220, 220),
                    1,
                    cv2.LINE_AA
                )

                writer.send(annotated.tobytes())
                frame_idx += 1

            return True

        except Exception as e:
            print(f"[VideoAnnotator] Error rendering annotated video for {view_name}: {e}")
            return False
        finally:
            cap.release()
            if writer:
                try:
                    writer.close()
                except Exception:
                    pass
