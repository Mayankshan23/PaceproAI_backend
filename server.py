"""
PacePro AI - Integrated Backend Server
======================================
FastAPI Server providing end-to-end integration with the PacePro React frontend:
1. GET  /health              - Backend status and diagnostics
2. POST /api/analyze-bowling - Multi-view bowling video analysis (Front, Side1, Side2)
3. POST /api/generate-diet   - Cricket fast-bowler dietary targets & meal plans
4. POST /api/generate-plan   - Combined fast bowling workload & synchronized nutrition plan
5. POST /api/workload        - ACWR (Acute:Chronic Workload Ratio) calculation
"""

import os
import sys
import shutil
import tempfile
import cv2
from typing import Optional, List, Dict, Any
from pathlib import Path

# Configure dynamic library search paths for MediaPipe / OpenCV on Linux / Render
_lib_search_dirs = [
    os.path.expanduser("~/.local/lib"),
    "/opt/render/project/src/.local/lib",
    "/opt/render/project/src/lib",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "lib")),
]
for _lp in _lib_search_dirs:
    if os.path.isdir(_lp):
        _cur_ld = os.environ.get("LD_LIBRARY_PATH", "")
        if _lp not in _cur_ld.split(":"):
            os.environ["LD_LIBRARY_PATH"] = f"{_lp}:{_cur_ld}" if _cur_ld else _lp

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import config
from pipeline import PaceProPipeline
from planner_service import PlannerService
from database import db_manager
from strength_diagnostics import strength_engine

app = FastAPI(
    title="PacePro AI Engine",
    description="Biomechanical pose estimation, ACWR workload, and synchronized athletic nutrition",
    version="1.0.0"
)

# Enable CORS for frontend Vite development and production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Locate models directory
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = str(BASE_DIR / "models" / "pose_landmarker_full.task")

planner_service = PlannerService()


# =====================================================================
# HEALTH CHECK & DB DIAGNOSTICS
# =====================================================================

@app.get("/health")
@app.get("/api/health")
async def health_check():
    model_exists = os.path.isfile(MODEL_PATH)
    fallbacks = [
        {"role": fb.get("role"), "provider": fb.get("provider"), "model": fb.get("model")}
        for fb in config.get_fallback_configs()
    ]
    return {
        "status": "ok",
        "service": "pacepro-ai",
        "version": "1.0.0",
        "model_loaded": model_exists,
        "model_path": MODEL_PATH,
        "llm_key_present": config.is_llm_configured(),
        "ai_provider": config.get_ai_provider(),
        "ai_model": config.get_ai_model_name(),
        "fallback_chain": fallbacks,
        "database": db_manager.get_status()
    }


# =====================================================================
# 1. BOWLING BIOMECHANICS ANALYZER ENDPOINT
# =====================================================================

METRIC_METADATA = {
    "back_knee_flexion_at_bfc": {
        "name": "Back Knee Flexion at BFC",
        "event": "BFC",
        "unit": "°",
        "optimal_range": "150° - 165°",
        "optimal_min": 150.0,
        "optimal_max": 165.0,
    },
    "front_knee_angle_at_ffc": {
        "name": "Front Knee Angle at FFC",
        "event": "FFC",
        "unit": "°",
        "optimal_range": "155° - 175°",
        "optimal_min": 150.0,
        "optimal_max": 178.0,
    },
    "front_knee_brace_change": {
        "name": "Front Knee Brace Change",
        "event": "FFC->Release",
        "unit": "°",
        "optimal_range": "15° - 22°",
        "optimal_min": 15.0,
        "optimal_max": 22.0,
    },
    "release_elbow_extension": {
        "name": "Bowling Arm Elbow at Release",
        "event": "Release",
        "unit": "°",
        "optimal_range": "165° - 180°",
        "optimal_min": 165.0,
        "optimal_max": 180.0,
    },
    "trunk_lean_at_release": {
        "name": "Trunk Lean at Release",
        "event": "Release",
        "unit": "°",
        "optimal_range": "8° - 18°",
        "optimal_min": 8.0,
        "optimal_max": 18.0,
    },
    "hip_shoulder_separation": {
        "name": "Hip-Shoulder Separation",
        "event": "FFC",
        "unit": "°",
        "optimal_range": "20° - 35°",
        "optimal_min": 20.0,
        "optimal_max": 35.0,
    },
    "arm_slot": {
        "name": "Arm Slot",
        "event": "Release",
        "unit": "°",
        "optimal_range": "High-Arm (60° - 90°)",
        "optimal_min": 60.0,
        "optimal_max": 90.0,
    }
}

def evaluate_metric_status(metric_key: str, val):
    if val is None:
        return "neutral"
    meta = METRIC_METADATA.get(metric_key)
    if not meta:
        return "optimal"
    if isinstance(val, (int, float)):
        min_val = meta.get("optimal_min", 0)
        max_val = meta.get("optimal_max", 180)
        if min_val <= val <= max_val:
            return "optimal"
        elif abs(val - min_val) <= 5 or abs(val - max_val) <= 5:
            return "good"
        else:
            return "warning"
    if isinstance(val, str):
        if "high" in val.lower():
            return "optimal"
        return "good"
    return "optimal"


def generate_fallback_coaching_report(measurements: Dict[str, Any], bowling_arm: str = "right"):
    """Generates an intelligent, encouraging, and positively framed biomechanical feedback report."""
    brace_val = measurements.get("front_knee_brace_change", {}).get("fused", 18.0)
    back_knee = measurements.get("back_knee_flexion_at_bfc", {}).get("fused", 154.0)
    elbow_ext = measurements.get("release_elbow_extension", {}).get("fused", 169.0)
    arm_slot_val = measurements.get("arm_slot", {}).get("fused", 72.0)
    trunk_lean = measurements.get("trunk_lean_at_release", {}).get("fused", 9.0)

    score = 86
    strengths = []
    opportunities = []
    recommendations = []

    if isinstance(brace_val, (int, float)) and brace_val >= 14:
        score += 4
        strengths.append(f"Outstanding lead leg firmness ({brace_val}° retention) acts as a powerful catapult fulcrum, transferring ground reaction force straight into release pace.")
    else:
        score -= 2
        opportunities.append(f"Prime Pace Opportunity (+3–5 km/h): Firming up your lead knee brace ({brace_val}° plant) will turn all your run-up power directly into extra ball speed!")
        recommendations.append("Perform single-leg landing block holds and wall lockouts to build solid lead leg stiffness.")

    if isinstance(back_knee, (int, float)) and back_knee >= 150:
        score += 3
        strengths.append(f"Tall springy gather at Back Foot Contact ({back_knee}°) smoothly carries run-up momentum into the delivery stride.")
    else:
        opportunities.append(f"Gather Flow Cue: Focusing on staying tall and springy on your back foot plant ({back_knee}°) will preserve full approach velocity.")
        recommendations.append("Bound-and-drive approach drills with cones to build a tall, rhythmic back-leg gather.")

    if isinstance(elbow_ext, (int, float)) and elbow_ext >= 165:
        strengths.append(f"Clean, fluid bowling arm extension ({elbow_ext}°) produces natural bounce and steep downward carry.")
    else:
        opportunities.append(f"High-Wheel Arc Focus: A long, relaxed bowling arm sweep ({elbow_ext}°) gives effortless whip and seam presentation.")
        recommendations.append("Towel whipping drills and high release arc target practice.")

    if isinstance(arm_slot_val, (int, float)) and arm_slot_val >= 65:
        strengths.append(f"High-arm release elevation ({arm_slot_val}°) maximizes steep bounce and upright seam presentation.")
    else:
        strengths.append(f"Natural three-quarter arm slot ({arm_slot_val}°) generates late swing and skid through the air.")

    if isinstance(trunk_lean, (int, float)) and trunk_lean > 15:
        opportunities.append(f"Upright Drive Cue: Driving your chest straight toward the target ({trunk_lean}° tilt) channels all your momentum directly at the stumps.")
        recommendations.append("Standing medicine ball chest throws (3 sets x 8 reps) and core rotational stability holds.")
    else:
        strengths.append("Upright thoracic drive minimizes rotation strain, keeping your back fresh for long match spells.")

    recommendations.append("Approach cadence run-up drills to groove a repeatable, effortless delivery rhythm.")

    grade = "A (Elite High-Performance Action)" if score >= 88 else "A- (Strong Kinetic Foundation)" if score >= 80 else "B+ (High-Potential Fast Bowler)"

    return {
        "overall_score": min(score, 96),
        "grade": grade,
        "priority_improvement": "Keep driving your chest tall toward the target at front-foot strike to convert your natural approach momentum into explosive release velocity.",
        "what_you_are_doing_well": strengths,
        "technical_issues": opportunities if opportunities else ["Sync arm delivery sweep with peak front leg lockout for effortless whip."],
        "coaching_recommendations": recommendations,
        "questions_to_discuss_with_coach": [
            "How can we groove the run-up cadence to hit the front-foot plant with maximum spring?",
            "Which release wrist cues give the most consistent upright seam on match days?"
        ]
    }


@app.post("/api/analyze-bowling")
async def analyze_bowling_api(
    front: Optional[UploadFile] = File(None),
    side1: Optional[UploadFile] = File(None),
    side2: Optional[UploadFile] = File(None),
    bowling_arm: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None),
    athlete_name: Optional[str] = Form(None),
):
    """
    Accepts video files for Front, Side 1, and Side 2 camera angles.
    Runs the multi-view PacePro pipeline and formats the response for the frontend UI.
    """
    video_files_dict = {}
    if front and front.filename:
        video_files_dict["front"] = front
    if side1 and side1.filename:
        video_files_dict["side1"] = side1
    if side2 and side2.filename:
        video_files_dict["side2"] = side2

    with tempfile.TemporaryDirectory() as temp_dir:
        video_paths = {}
        for view_key, upload in video_files_dict.items():
            # Filter out non-file strings that might have been sent by frontend
            if not hasattr(upload, "filename") or not upload.filename:
                continue
            clean_filename = f"{view_key}_{upload.filename}"
            save_path = os.path.join(temp_dir, clean_filename)
            with open(save_path, "wb") as f:
                shutil.copyfileobj(upload.file, f)

            # Strict 11-second video duration validation
            v_cap = cv2.VideoCapture(save_path)
            if v_cap.isOpened():
                v_fps = v_cap.get(cv2.CAP_PROP_FPS)
                v_frames = v_cap.get(cv2.CAP_PROP_FRAME_COUNT)
                v_cap.release()
                if v_fps and v_fps > 0:
                    v_duration = v_frames / v_fps
                    if v_duration > 11.5:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Video for '{view_key}' view is {v_duration:.1f} seconds, exceeding the 11-second maximum limit. Please upload a clip trimmed to <= 11s."
                        )

            video_paths[view_key] = save_path

        # The analysis pipeline requires at least one uploaded camera view.
        if not video_paths:
            raise HTTPException(status_code=400, detail="Upload at least one bowling video view.")

        # Normalize bowling arm hint from athlete profile or UI if provided
        arm_hint = None
        if bowling_arm:
            b_clean = str(bowling_arm).strip().lower()
            if "left" in b_clean:
                arm_hint = "left"
            elif "right" in b_clean:
                arm_hint = "right"

        # Run real CV Pipeline
        pipeline = PaceProPipeline(pose_model_path=MODEL_PATH)
        has_llm = config.is_llm_configured()
        raw_result = pipeline.analyze_bowling(video_paths, run_llm=has_llm, bowling_arm_hint=arm_hint)

        # Adapt pipeline output to the frontend format
        views_used = raw_result.get("views_used", [])
        views_missing = raw_result.get("views_missing_or_failed", [])
        per_view_events = {}
        for v_name, ev_data in raw_result.get("final_measurements", {}).get("per_view_events", {}).items():
            if ev_data.get("success"):
                per_view_events[v_name] = {
                    "bfc_frame": ev_data.get("bfc", {}).get("frame"),
                    "ffc_frame": ev_data.get("ffc", {}).get("frame"),
                    "release_frame": ev_data.get("release_frame")
                }

        # Format measurements with per-view values, confidence %, and fusion status
        fusion_payload = raw_result.get("final_measurements", {}).get("view_fusion", {})
        raw_measurements = fusion_payload.get("measurements", {})
        formatted_measurements = {}

        for key, meta in METRIC_METADATA.items():
            meas = raw_measurements.get(key, {})
            fused_val = meas.get("value")
            
            # Non-numeric fallback for arm_slot if string returned
            if key == "arm_slot" and isinstance(fused_val, str):
                slot_angle = 75.0 if "high" in fused_val else 45.0 if "three" in fused_val else 20.0
                display_fused = slot_angle
            else:
                display_fused = fused_val if fused_val is not None else 0.0

            status = evaluate_metric_status(key, display_fused)

            cand_dict = meas.get("candidates", {})
            front_val = meas.get("front", cand_dict.get("front"))
            side1_val = meas.get("side1", cand_dict.get("side1"))
            side2_val = meas.get("side2", cand_dict.get("side2"))

            formatted_measurements[key] = {
                "name": meta["name"],
                "event": meas.get("event") or meta.get("event", "Event"),
                "definition": meas.get("definition", ""),
                "front": front_val,
                "side1": side1_val,
                "side2": side2_val,
                "fused": display_fused,
                "confidence_pct": meas.get("confidence_pct", 38 if len(views_used) == 1 else 85),
                "confidence_label": meas.get("confidence", "low" if len(views_used) == 1 else "high"),
                "fusion_status": meas.get("status", "single_view_low_confidence" if len(views_used) == 1 else "consensus"),
                "selected_views": meas.get("selected_views", views_used),
                "rejected_outliers": meas.get("rejected_outliers", []),
                "unit": meta["unit"],
                "optimal_range": meta["optimal_range"],
                "status": status,
                "note": meas.get("note", "")
            }

        # Coaching report
        bowling_arm = raw_result.get("final_measurements", {}).get("bowling_arm")
        if arm_hint:
            bowling_arm_str = arm_hint
        elif isinstance(bowling_arm, dict):
            bowling_arm_str = bowling_arm.get("value", "right")
        else:
            bowling_arm_str = str(bowling_arm) if bowling_arm else "right"

        llm_report = raw_result.get("llm_report")
        if not llm_report or llm_report.get("_status") == "unavailable":
            llm_report = generate_fallback_coaching_report(formatted_measurements, bowling_arm_str)
        else:
            # Map fields if LLM used alternative key names
            if "strengths" in llm_report and "what_you_are_doing_well" not in llm_report:
                llm_report["what_you_are_doing_well"] = llm_report.get("strengths", [])
            if "possible_issues" in llm_report and "technical_issues" not in llm_report:
                llm_report["technical_issues"] = llm_report.get("possible_issues", [])
            if "recommendations" in llm_report and "coaching_recommendations" not in llm_report:
                llm_report["coaching_recommendations"] = llm_report.get("recommendations", [])
            if "questions_to_discuss_with_coach" not in llm_report:
                llm_report["questions_to_discuss_with_coach"] = [
                    "How can we maintain arm speed across long match spells without strain?",
                    "Should approach foot plants be slightly shortened for better knee brace alignment?"
                ]
            if "grade" not in llm_report:
                llm_report["grade"] = "A- (High Performance Candidate)"
            if "priority_improvement" not in llm_report:
                llm_report["priority_improvement"] = llm_report.get("summary", "Maintain firm front knee brace at foot contact.")
            if "overall_score" not in llm_report or not isinstance(llm_report.get("overall_score"), (int, float)):
                calc_score = 82
                brace = formatted_measurements.get("front_knee_brace_change", {}).get("fused", 0)
                if isinstance(brace, (int, float)) and brace >= 0:
                    calc_score += 5
                elif isinstance(brace, (int, float)) and brace < -10:
                    calc_score -= 6
                elbow = formatted_measurements.get("release_elbow_extension", {}).get("fused", 170)
                if isinstance(elbow, (int, float)) and elbow >= 165:
                    calc_score += 4
                elif isinstance(elbow, (int, float)) and elbow < 155:
                    calc_score -= 8
                trunk = formatted_measurements.get("trunk_lean_at_release", {}).get("fused", 12)
                if isinstance(trunk, (int, float)) and trunk <= 18:
                    calc_score += 4
                elif isinstance(trunk, (int, float)) and trunk > 22:
                    calc_score -= 6
                llm_report["overall_score"] = min(96, max(50, calc_score))

        single_view_warning = fusion_payload.get("single_view_warning")
        overall_confidence_pct = fusion_payload.get("overall_confidence_pct", 38 if len(views_used) == 1 else 85)
        disagreement_detected = fusion_payload.get("disagreement_detected", False)

        final_res = {
            "pipeline_status": "success" if views_used else "no_usable_views",
            "views_requested": raw_result.get("views_requested", list(video_paths.keys())),
            "views_used": views_used,
            "views_missing_or_failed": views_missing,
            "view_errors": raw_result.get("view_errors", {}),
            "single_view_warning": single_view_warning,
            "overall_confidence_pct": overall_confidence_pct,
            "disagreement_detected": disagreement_detected,
            "final_measurements": {
                "per_view_events": per_view_events,
                "bowling_arm": bowling_arm_str,
                "overall_confidence_pct": overall_confidence_pct,
                "single_view_warning": single_view_warning,
                "disagreement_detected": disagreement_detected,
                "view_fusion": {
                    "views_used": views_used,
                    "views_missing_or_failed": views_missing,
                    "overall_confidence_pct": overall_confidence_pct,
                    "single_view_warning": single_view_warning,
                    "disagreement_detected": disagreement_detected,
                    "measurements": formatted_measurements
                }
            },
            "llm_report": llm_report
        }

        # Auto-persist analysis record and history item to MongoDB
        if db_manager.is_connected:
            uid = user_id or "default_athlete"
            db_record = {k: v for k, v in final_res.items()}
            db_record["user_id"] = uid
            db_manager.save_analysis_record(db_record, user_id=uid, max_retained=3)
            brace_val = formatted_measurements.get("front_knee_brace_change", {}).get("fused", "N/A")
            db_manager.save_history_record({
                "user_id": uid,
                "type": "bowling",
                "title": f"Multi-View Bowling Biomechanics ({athlete_name or 'Fast Bowler'})",
                "subtitle": f"{len(views_used)} views analyzed • Brace angle {brace_val}°",
                "status": "Completed",
                "detail": db_record
            }, user_id=uid, max_retained=3)

        return final_res



# =====================================================================
# 2. ATHLETE DIETARY PLANNER ENDPOINT
# =====================================================================

class DietPlanRequest(BaseModel):
    user_id: Optional[str] = "default_athlete"
    age: int = 20
    weight: float = 70.0
    height: float = 178.0
    sex: str = "male"
    activity: str = "high"
    goal: str = "performance"
    food_preference: Optional[str] = None
    foodPref: Optional[str] = None
    budgetWeekly: Optional[float] = 3500.0
    allergies: Optional[List[str]] = []
    avoidFoods: Optional[List[str]] = []
    mealFreq: Optional[int] = 5
    cookingPref: Optional[str] = "home"
    recoveryNote: Optional[str] = "none"
    calculatedBmr: Optional[float] = None
    calculatedTdee: Optional[float] = None
    calculatedCalories: Optional[float] = None

    class Config:
        extra = "allow"


@app.post("/api/generate-diet")
def generate_diet_api(req: DietPlanRequest):
    """
    Computes daily dietary targets using Mifflin-St Jeor BMR and generates
    cricket-specific daily athletic meal breakdowns.
    """
    try:
        result = planner_service.generate_diet_plan(req.dict())
        if db_manager.is_connected:
            uid = req.user_id or "default_athlete"
            db_manager.save_plan_record("diet", result, user_id=uid, max_retained=3)
            targets = result.get("daily_targets", {})
            db_manager.save_history_record({
                "user_id": uid,
                "type": "diet",
                "title": "Athlete Nutrition Plan",
                "subtitle": f"{targets.get('calories', 2800)} kcal • {targets.get('protein_g', 150)}g Protein",
                "status": "Active",
                "detail": result
            }, user_id=uid, max_retained=3)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# =====================================================================
# 3. COMBINED WORKLOAD & NUTRITION PLANNER ENDPOINT
# =====================================================================

@app.post("/api/generate-plan")
def generate_combined_plan_api(payload: Dict[str, Any]):
    """
    Generates a synchronized 7-day cricket fast-bowling workload & nutrition plan
    adapting to Student vs Full-Time Athlete status, gym volume, ground volume,
    and max-overs spells with ACWR injury risk monitoring.
    """
    try:
        result = planner_service.generate_combined_plan(payload)
        if db_manager.is_connected:
            uid = payload.get("user_id") or "default_athlete"
            db_manager.save_plan_record("combined", result, user_id=uid, max_retained=3)
            db_manager.save_history_record({
                "user_id": uid,
                "type": "combined_plan",
                "title": result.get("title", "Fast Bowler Unified Workload & Nutrition Protocol"),
                "subtitle": f"{result.get('workload_summary', {}).get('gym_days', 3)} Gym • {result.get('workload_summary', {}).get('ground_days', 4)} Ground • {result.get('nutrition_summary', {}).get('training_day_calories', 2800)} kcal",
                "status": "Active Protocol",
                "detail": result
            }, user_id=uid, max_retained=3)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/generate-workout")
def generate_workout_api(payload: Dict[str, Any]):
    """Generates a seven-day workout schedule for the training dashboard."""
    try:
        result = planner_service.generate_workout_plan(payload)
        if db_manager.is_connected:
            uid = payload.get("user_id") or "default_athlete"
            db_manager.save_plan_record("workout", result, user_id=uid, max_retained=3)
            db_manager.save_history_record({
                "user_id": uid,
                "type": "workout",
                "title": "7-Day Fast Bowler Workout Protocol",
                "subtitle": f"{result.get('days_count', 7)} Days Scheduled • ACWR Monitored",
                "status": "Active",
                "detail": result
            }, user_id=uid, max_retained=3)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# =====================================================================
# 4. WORKLOAD ACWR ENDPOINT
# =====================================================================

class WorkloadRequest(BaseModel):
    sessions: List[Dict[str, Any]]
    reference_date: Optional[str] = None


@app.post("/api/workload")
async def calculate_workload_api(req: WorkloadRequest):
    """
    Computes Acute:Chronic Workload Ratio from a history of training sessions.
    """
    try:
        result = planner_service.calculate_workload(req.sessions, reference_date=req.reference_date)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# =====================================================================
# 4B. FAST BOWLER STRENGTH & MUSCLE WEAKNESS ASSESSMENT ENDPOINT
# =====================================================================

class StrengthAssessmentRequest(BaseModel):
    user_id: Optional[str] = "default_athlete"
    weight_kg: Optional[float] = 68.0
    age: Optional[int] = 20
    strength_track: Optional[str] = "gym_lifting"
    bowling_arm: Optional[str] = "Right-Arm Fast"
    bowling_speed_kph: Optional[float] = 132.0
    split_squat_kg: Optional[float] = 14.0
    split_squat_reps: Optional[int] = 8
    wall_sit_seconds: Optional[float] = 40.0
    does_deadlifts: Optional[bool] = True
    deadlift_kg: Optional[float] = 85.0
    deadlift_reps: Optional[int] = 5
    bodyweight_squats_reps: Optional[int] = 25
    plank_seconds: Optional[float] = 60.0
    side_plank_seconds: Optional[float] = 40.0
    pullups_count: Optional[int] = 6
    pushups_reps: Optional[int] = 25
    calf_raises_reps: Optional[int] = 18
    reported_niggle: Optional[str] = "none"


@app.post("/api/assess-strength")
def assess_strength_api(req: StrengthAssessmentRequest):
    """
    Evaluates bowler physical strength across 5 fast-bowling kinetic chain links,
    identifies primary muscle weakness and injury risks, and prescribes targeted corrective exercises.
    """
    try:
        data = req.dict()
        result = strength_engine.analyze_bowler_strength(data)
        
        # Persist to MongoDB if connected
        if db_manager.is_connected:
            uid = req.user_id or "default_athlete"
            db_manager.save_plan_record("strength_assessment", result, user_id=uid, max_retained=3)
            db_manager.save_history_record({
                "user_id": uid,
                "type": "strength_test",
                "title": f"Strength Test: {result['primary_weakness']['name']}",
                "subtitle": f"Score {result['kinetic_chain_score']}/100 • Weakness: {result['primary_weakness']['affected_muscles'].split(',')[0]}",
                "status": result["overall_rating"],
                "detail": result
            }, user_id=uid, max_retained=3)
            
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# =====================================================================
# 5. MONGODB ATLAS ATHLETE PROFILE & RECORDS ENDPOINTS
# =====================================================================

@app.get("/api/db/status")
async def get_db_status():
    """Returns real-time status of the MongoDB database connection."""
    return db_manager.get_status()


class AthleteProfilePayload(BaseModel):
    user_id: str
    profile: Dict[str, Any]


@app.post("/api/athlete-profile")
async def save_profile_endpoint(payload: AthleteProfilePayload):
    """Saves or updates athlete profile in MongoDB."""
    ok = db_manager.save_athlete_profile(payload.user_id, payload.profile)
    return {"success": ok, "connected": db_manager.is_connected}


@app.get("/api/athlete-profile/{user_id}")
async def get_profile_endpoint(user_id: str):
    """Retrieves athlete profile from MongoDB."""
    doc = db_manager.get_athlete_profile(user_id)
    if not doc:
        return {"found": False, "profile": None}
    return {"found": True, "profile": doc}


# =====================================================================
# 6. HISTORY TIMELINE ENDPOINTS (Auto-Deletion: Retaining Latest 3 Generations)
# =====================================================================

class HistoryItemPayload(BaseModel):
    id: Optional[str] = None
    user_id: Optional[str] = "default_athlete"
    type: str = "general"
    title: str = "Performance Log"
    subtitle: Optional[str] = ""
    status: Optional[str] = "Active"
    date: Optional[str] = None
    detail: Optional[Dict[str, Any]] = None
    max_retained: Optional[int] = 3


@app.get("/api/history")
async def get_history_endpoint(user_id: Optional[str] = None, type: Optional[str] = None, limit: int = 30):
    """Retrieves athlete history records from MongoDB."""
    items = db_manager.get_history(user_id=user_id, item_type=type, limit=limit)
    return {"status": "ok", "count": len(items), "history": items}


@app.post("/api/history")
async def save_history_endpoint(item: HistoryItemPayload):
    """
    Saves a history item to MongoDB and auto-deletes older generations beyond 3 items
    for that athlete and category.
    """
    res = db_manager.save_history_record(
        item.dict(),
        user_id=item.user_id,
        max_retained=item.max_retained or 3
    )
    updated_list = db_manager.get_history(user_id=item.user_id)
    return {"status": "ok", "saved_info": res, "history": updated_list}


@app.delete("/api/history/{history_id}")
async def delete_history_endpoint(history_id: str, user_id: Optional[str] = None):
    """Deletes a specific history record from MongoDB."""
    ok = db_manager.delete_history_item(history_id, user_id=user_id)
    return {"status": "ok", "deleted": ok}


# =====================================================================
# 6. CONTEXT-AWARE AI COACH CHAT ENDPOINT (HINGLISH & ENGLISH)
# =====================================================================

class ChatRequest(BaseModel):
    message: str
    language: Optional[str] = "hinglish"  # "hinglish" | "english"
    history: Optional[List[Dict[str, Any]]] = []
    athlete_profile: Optional[Dict[str, Any]] = None
    latest_analysis: Optional[Dict[str, Any]] = None
    active_diet: Optional[Dict[str, Any]] = None
    active_workout: Optional[Dict[str, Any]] = None
    active_combined_plan: Optional[Dict[str, Any]] = None


@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    """
    Connected, empathetic, and context-aware AI Coach giving authentic cricket mentorship.
    Supports natural locker-room Hinglish and pure English.
    """
    # 1. Build rich athlete context summary
    context_lines = []

    athlete_name = "Bowler"
    first_name = "Champ"
    if req.athlete_profile:
        p = req.athlete_profile
        athlete_name = p.get("name", "Bowler")
        first_name = athlete_name.split()[0] if athlete_name else "Champ"
        arm_text = "Left-Arm Fast (LAF)" if p.get("bowlingArm") == "left" or p.get("arm") == "left" else "Right-Arm Fast (RAF)"
        action_text = (p.get("bowlingAction") or "Semi-Open").replace("_", " ").title()
        context_lines.append(
            f"ATHLETE PROFILE: {athlete_name} | {arm_text} | Action: {action_text} | "
            f"Goal: {p.get('goal', 'increase_speed')} | Weight: {p.get('weight', 68)}kg | Age: {p.get('age', 20)}y"
        )
        if p.get("sleepHours"):
            context_lines.append(f"RECOVERY BASELINE: Sleep: {p.get('sleepHours')}h/night | Water: {p.get('waterIntake', 3.5)}L/day")

    has_analysis = bool(req.latest_analysis)
    knee_brace_val = "18.0"
    trunk_flex_val = "9.0"
    elbow_ext_val = "169.0"
    arm_slot_val = "72.0"
    report_grade = "A- (Elite Candidate)"
    report_score = 88
    priority_fix = "Maintain firm front knee brace at foot contact."
    has_skeleton_video = False

    if req.latest_analysis:
        a = req.latest_analysis
        fusion = a.get("final_measurements", {}).get("view_fusion", {}).get("measurements", {})
        knee_brace_val = fusion.get("front_knee_brace_change", {}).get("fused", 18.0)
        trunk_flex_val = fusion.get("trunk_lean_at_release", {}).get("fused", 9.0)
        elbow_ext_val = fusion.get("release_elbow_extension", {}).get("fused", 169.0)
        arm_slot_val = fusion.get("arm_slot", {}).get("fused", 72.0)
        report = a.get("llm_report", {})
        report_grade = report.get("grade", "A- (Elite Candidate)")
        report_score = report.get("overall_score", 88)
        priority_fix = report.get("priority_improvement", "Maintain firm front knee brace at foot contact.")
        has_skeleton_video = bool(a.get("annotated_videos"))
        context_lines.append(
            f"PREVIOUS VIDEO ANALYSIS ON RECORD (YOU ALREADY ANALYZED THIS ATHLETE'S VIDEO):\n"
            f"  - Status: COMPLETED (Score: {report_score}/100, Grade: {report_grade})\n"
            f"  - Front Knee Brace Retention: {knee_brace_val}°\n"
            f"  - Bowling Arm Elbow: {elbow_ext_val}°\n"
            f"  - Trunk Lean at Release: {trunk_flex_val}°\n"
            f"  - Arm Slot: {arm_slot_val}°\n"
            f"  - Priority Coaching Correction: '{priority_fix}'\n"
            f"  - 3D Skeleton Video: {'Ready to watch/download' if has_skeleton_video else 'Generated'}"
        )
    else:
        context_lines.append("VIDEO ANALYSIS STATUS: No video analyzed yet in current session.")

    if req.active_diet:
        targets = req.active_diet.get("daily_targets", {})
        context_lines.append(f"DIET TARGETS: {targets.get('calories', 2800)} kcal, {targets.get('protein_g', 150)}g Protein")

    if req.active_workout:
        context_lines.append(f"WORKOUT: Focus {req.active_workout.get('focus', 'Speed & Lumbar Protection')}")

    if req.active_combined_plan:
        c = req.active_combined_plan
        ws = c.get("workload_summary", {})
        context_lines.append(
            f"WORKLOAD PROTOCOL: ACWR {ws.get('acwr_score', 1.15)} ({ws.get('acwr_status', 'Sweet Spot')}), "
            f"{ws.get('ground_days', 4)} Ground ({ws.get('total_weekly_overs', 16)} overs/wk) / {ws.get('gym_days', 3)} Gym Days"
        )

    context_str = "\n".join(context_lines) if context_lines else f"Athlete Name: {athlete_name}. No detailed analysis generated yet."

    # Language & Persona Tone
    lang = (req.language or "hinglish").lower().strip()
    if lang == "english":
        lang_instruction = (
            f"LANGUAGE & TONE: Inspiring, authentic, direct coaching ENGLISH. "
            f"Address the bowler as '{first_name}'. Talk like an elite fast bowling mentor at the bowling crease. "
            f"LENGTH CONSTRAINT (CRITICAL): Keep replies SHORT, PUNCHY, and CONVERSATIONAL (2 to 4 sentences, ~40-60 words max). "
            f"DO NOT write long essays or unasked bullet points. Use long text ONLY if {first_name} explicitly asks to 'explain in detail' or 'give full roadmap'."
        )
    else:
        lang_instruction = (
            f"LANGUAGE & TONE: Authentic, motivating Cricket Locker-Room HINGLISH (natural Hindi + English in Roman script). "
            f"Address the bowler warmly as '{first_name} bhai' or 'Champ'. Speak like an elder brother and head coach at the nets "
            f"(jaise Bumrah, Shami, Wasim Akram dressing room me baat karte hain). "
            f"LENGTH CONSTRAINT (CRITICAL): Keep replies SHORT, PUNCHY, and DIRECT (2 to 4 crisp sentences, ~40-60 words max). "
            f"Zabardasti lamba essay ya bullet list mat likho. Lamba detailed jawab SIRF tabhi do jab user clearly boley 'detail me batao' ya 'step by step roadmap do'."
        )

    system_prompt = f"""You are 'Coach PacePro', the personal, trusted Cricket Head Coach and Fast Bowling Biomechanist to {athlete_name}.

YOUR CRICKET EXPERTISE:
You can answer any question about cricket trials/selections (DDCA, MCA, KSCA, UPCA, BCA, MRF Pace Foundation), bowling mechanics (lead knee brace, trunk alignment, arm slot), ball craft (outswing, inswing, wobble seam, reverse swing, yorkers), and recovery.

ATHLETE'S PERSONAL DATA ON RECORD:
{context_str}

CRITICAL RULES:
1. CONCISE & PUNCHY BY DEFAULT: Keep your response short (2 to 4 sentences). Avoid long lectures. Use bullet points ONLY if the user explicitly asks for a full breakdown/roadmap.
2. PREVIOUS VIDEO RECALL: If {first_name} asks about their analyzed video, proudly recall their score ({report_score}/100, {report_grade}) and knee brace ({knee_brace_val}°), and tell them to click the report button below.
3. {lang_instruction}"""

    # Conversation History
    convo_history = []
    for msg in (req.history or [])[-4:]:
        role = "Athlete" if msg.get("sender") == "user" else "Coach"
        convo_history.append(f"{role}: {msg.get('text', '')}")

    user_query = f"{first_name} asks: {req.message}"
    if convo_history:
        user_prompt = "Recent chat:\n" + "\n".join(convo_history) + f"\n\n{first_name} asks: {req.message}"
    else:
        user_prompt = user_query

    # Determine actionable CTA buttons based on query
    action_suggestions = []
    low_msg = req.message.lower()

    # Detect if user is asking about an ALREADY analyzed video
    is_recalling_analysis = any(w in low_msg for w in [
        "tunai", "tune", "kri thi", "kari thi", "pehle", "already", "dekha tha", "dekh liya",
        "purani", "result", "mera score", "kaisa tha", "kya aya", "kya aaya", "pta hai", "pata hai", "yaad hai"
    ]) and any(w in low_msg for w in ["video", "analyse", "analyze", "analysis", "action", "report", "brace"])

    if has_analysis and is_recalling_analysis:
        action_suggestions.append({
            "id": "view_analysis",
            "type": "navigate",
            "tab": "analyzer",
            "label": "📊 Meri Analysis Report & Skeleton Video Kholo" if lang == "hinglish" else "📊 View My Analysis Report & Video",
            "primary": True
        })
    elif any(w in low_msg for w in ["video", "analyse", "analyze", "analysis", "action", "clip", "angle", "brace", "stride", "upload"]) and not any(w in low_msg for w in ["trial", "selection"]):
        action_suggestions.append({
            "id": "go_analyzer",
            "type": "navigate",
            "tab": "analyzer",
            "label": "🎥 Bowling Video Analyzer Kholo" if lang == "hinglish" else "🎥 Open Video Analyzer",
            "primary": True
        })

    if any(w in low_msg for w in ["diet", "nutrition", "protein", "meal", "khana", "chart", "calories", "macro", "bnwa", "bana", "khao", "food"]):
        action_suggestions.append({
            "id": "go_diet",
            "type": "navigate",
            "tab": "diet",
            "label": "🥗 Open Diet Planner" if lang == "english" else "🥗 Diet Planner Kholo",
            "primary": True
        })
        action_suggestions.append({
            "id": "go_training",
            "type": "navigate",
            "tab": "training",
            "label": "⚡ Workload & Fueling Sync" if lang == "english" else "⚡ Combined Workload & Diet",
            "primary": False
        })

    if any(w in low_msg for w in ["workout", "workload", "gym", "exercise", "over", "spell", "recovery", "schedule"]) and not any(w in low_msg for w in ["diet", "food", "khana"]):
        action_suggestions.append({
            "id": "go_workout",
            "type": "navigate",
            "tab": "workout",
            "label": "🏋️ 7-Day Bowling Schedule" if lang == "english" else "🏋️ 7-Day Workout Schedule Kholo",
            "primary": True
        })

    # LLM execution with smart fallback
    try:
        from llm_analyzer import LLMClient
        llm = LLMClient(max_tokens=400)
        reply_text = llm.generate_text(system_prompt, user_prompt)
        return {
            "reply": reply_text.strip(),
            "language": lang,
            "has_context": bool(context_lines),
            "action_suggestions": action_suggestions
        }
    except Exception as e:
        # Dynamic, warm, contextual coach fallback based on user query
        low_msg = req.message.lower()
        prot_target = 150
        if req.active_diet:
            prot_target = req.active_diet.get("daily_targets", {}).get("protein_g", 150)
        elif req.athlete_profile:
            prot_target = int(round(req.athlete_profile.get("weight", 68) * 2.2))

        # 1. Video Analysis Recall Fallback
        if has_analysis and is_recalling_analysis:
            if lang == "english":
                fallback = (
                    f"Of course I remember, {first_name}! We already analyzed your delivery stride: "
                    f"your front knee brace retention was {knee_brace_val}°, release elbow at {elbow_ext_val}°, with an overall score of {report_score}/100 ({report_grade}). "
                    f"Our main focus was: '{priority_fix}'. Click the button below to review your report and watch your skeleton video!"
                )
            else:
                fallback = (
                    f"Arre {first_name} bhai, 100% yaad hai! Humne jo delivery clip analyze ki thi usme tera front-knee brace {knee_brace_val}° tha, "
                    f"bowling elbow {elbow_ext_val}° aur overall grade {report_grade} ({report_score}/100) aaya tha. "
                    f"Tera main focus point tha: '{priority_fix}'. Neeche button pe click karke apni full report aur skeleton video dobara dekh le!"
                )
        # 2. Cricket Trials & Selection Process
        elif any(w in low_msg for w in ["trial", "trials", "selection", "academy", "district", "state team", "ranji", "u19", "u-19", "u16", "u-16", "u23", "mrf", "bcci"]):
            if lang == "english":
                fallback = (
                    f"Here is your direct roadmap for Cricket Trials & State Selection, {first_name}:\n\n"
                    f"1. **District Cricket Association (DCA) Trials**: Every state association (DDCA, MCA, KSCA, UPCA, TNCA, etc.) conducts open district trials between **July and September** before the domestic season.\n"
                    f"2. **Age-Group Pathways**: Register for U-16 (Vijay Merchant), U-19 (Vinoo Mankad / Cooch Behar), or U-23 (CK Nayudu) through your district association's official portal.\n"
                    f"3. **What Selectors Look For**:\n"
                    f"   - **Repeatable Pace & Line**: Bowling 6 balls on good length around the 4th stump.\n"
                    f"   - **Clean Action**: Firm front knee brace and smooth run-up without no-balls.\n"
                    f"   - **Fitness Baseline**: Yo-Yo test score 16.5+ and sprint stamina.\n"
                    f"4. **Open Fast Bowling Scouts**: Keep an eye on MRF Pace Foundation (Chennai) annual trials and IPL franchise open scouting.\n\n"
                    f"Which city or state are you from? Let me know your age group so I can give you the exact district association registration details!"
                )
            else:
                fallback = (
                    f"Sun {first_name} bhai, Cricket Trials aur State Selection ka solid roadmap yeh raha:\n\n"
                    f"1. **District Cricket Association (DCA) Trials**: Har State Association (jaise DDCA, MCA, UPCA, PCA, CAB, KSCA) domestic season se pehle (**July se September** ke beech) open district trials conduct karti hai.\n"
                    f"2. **Age-Group Tournament Pathway**:\n"
                    f"   - **U-16**: Vijay Merchant Trophy\n"
                    f"   - **U-19**: Vinoo Mankad (One-Day) & Cooch Behar Trophy (4-Day)\n"
                    f"   - **U-23**: Col CK Nayudu Trophy\n"
                    f"   - **Senior**: Ranji Trophy, Syed Mushtaq Ali & Vijay Hazare\n"
                    f"3. **Selectors Trials me kya dekhte hain?**:\n"
                    f"   - **Consistent Line & Length**: 6 balls lagatar off-stump ke bahar good length pe hit karna.\n"
                    f"   - **Clean Delivery Stride**: Front-knee brace firm ho aur run-up me zero no-balls.\n"
                    f"   - **Fitness & Body Language**: Energetic run-up aur high release point.\n"
                    f"4. **Fast Bowler Talent Hunts**: MRF Pace Foundation (Chennai) aur IPL Academy trials (Rajasthan Royals, MI grassroots) ke annual trials pe nazar rakho.\n\n"
                    f"Bata tu kis City/State se belong karta hai aur age category kya hai? Main tere state association ki exact details bata dunga!"
                )
        # 3. Knee Brace & Action Mechanics
        elif any(w in low_msg for w in ["knee", "brace", "plant", "lock"]):
            if lang == "english":
                fallback = (
                    f"Listen {first_name}, that front-knee brace is your primary velocity lever! "
                    f"At delivery stride, think of planting your lead foot like an iron pillar—don't let it buckle. "
                    f"Hit 3 sets of isometric wall-sits (45s holds) after your session to build lead-leg stiffness. Let's lock it in!"
                )
            else:
                fallback = (
                    f"Arre {first_name} bhai, front-knee brace bowling speed ka sabse bada lever hai! "
                    f"Delivery crease pe lead leg ko iron pillar ki tarah firm lock rakho, taaki sari ground force ball me transfer ho. "
                    f"Aaj gym me 3 sets isometric wall-sits lagao. Practice se knee brace ekdum solid ho jayega!"
                )
        # 4. Ball Craft / Swing / Variations
        elif any(w in low_msg for w in ["swing", "outswing", "inswing", "reverse", "seam", "wobble", "cutter", "knuckle", "yorker", "bouncer"]):
            if lang == "english":
                fallback = (
                    f"Here is the secret to mastering seam & swing, {first_name}:\n\n"
                    f"- **Outswing**: Keep the seam angled at 1st slip, wrist locked behind the ball, shiny side on the left (for RHB).\n"
                    f"- **Inswing**: Angle seam towards fine leg, drive the thumb through the crease with a slightly higher wrist release.\n"
                    f"- **Wobble Seam**: Hold the ball across the seam lightly with fingers split—creates unpredictable deviation off the pitch like Shami and Anderson.\n"
                    f"- **Reverse Swing**: Keep one side rough, completely dry, and bowl with maximum pace into the rough side airflow.\n\n"
                    f"Which variation are you sharpening in the nets this week?"
                )
            else:
                fallback = (
                    f"{first_name} bhai, swing aur variations ka masterclass yeh raha:\n\n"
                    f"- **Outswing**: Seam ko 1st slip ki taraf angle karo, wrist locked rakho aur shiny side left me rakho (for RHB).\n"
                    f"- **Inswing**: Seam ko fine leg ki taraf tilt karo aur release ke waqt thumb se seam ko drive karo.\n"
                    f"- **Wobble Seam (Shami style)**: Seam ko thoda loose pakdo fingers split karke—ball pitch pe drop hoke dono taraf jagah banayegi.\n"
                    f"- **Reverse Swing**: Ball ki ek side ko dry aur rough rakho, aur old ball se full pace se bowl karo.\n\n"
                    f"Bata aaj nets me outswing pe focus hai ya yorker execution pe?"
                )
        # 5. Diet & Nutrition
        elif any(w in low_msg for w in ["diet", "food", "protein", "khane", "recovery", "meal", "calorie"]):
            if lang == "english":
                fallback = (
                    f"{first_name}, fast bowling demands high-octane fueling! "
                    f"Take 40g of fast carbs (banana with honey toast) 45 min before bowling spells for explosive kinetic power. "
                    f"Hit your {prot_target}g daily protein target post-session to rebuild tendon collagen and prevent stress fractures. Eat like a champion!"
                )
            else:
                fallback = (
                    f"{first_name} bhai, bowling me pace tabhi aayegi jab body sahi se fuel hogi! "
                    f"Match spell se 45 min pehle fast carbs (banana + honey toast) lo taaki overs 5-8 me stamina bani rahe. "
                    f"Aur session ke baad apna {prot_target}g protein target pura karo. Recovery me koi compromise nahi!"
                )
        # 6. Pace & Velocity
        elif any(w in low_msg for w in ["speed", "pace", "130", "140", "tez", "velocity"]):
            if lang == "english":
                fallback = (
                    f"Speed comes from run-up rhythm and explosive front-leg plant, {first_name}! "
                    f"Smoothly accelerate over your first 10 strides, build momentum, and snap your thoracic spine over a stiff lead leg. "
                    f"Stay tall at the crease—that 130+ km/h pace will unlock naturally without forcing your arm."
                )
            else:
                fallback = (
                    f"Speed arm force se nahi, run-up rhythm aur front-foot plant se aati hai {first_name} bhai! "
                    f"Run-up me pehle 10 paces smoothly accelerate karo, jump over nahi balki bound through crease karo. "
                    f"Front knee ko lock rakhoge to 130+ km/h velocity naturally niklegi. Poora trust rakho!"
                )
        # 7. Spine & Injury Safety
        elif any(w in low_msg for w in ["back", "spine", "dard", "pain", "niggle", "lumbar", "injury"]):
            if lang == "english":
                fallback = (
                    f"Spinal health is our number one priority, {first_name}! "
                    f"Watch out for counter-rotation between hips and shoulders at back-foot landing. "
                    f"Always maintain at least 48 hours between heavy bowling spells, and do rotational core anti-flexion drills today. Let me know how it feels."
                )
            else:
                fallback = (
                    f"{first_name} bhai, spinal health priority #1 hai! Spine pe koi extra risk nahi lena. "
                    f"Ensure karo ki back-foot landing pe shoulders hips ke opposite na twist ho rahe hon. "
                    f"Heavy bowling spells ke beech strict 48 hours ka gap rakho aur core ko rest do. Niggle thoda bhi lage to bowling roko!"
                )
        # 8. General Cricket Coaching Fallback
        else:
            if lang == "english":
                fallback = (
                    f"Great to have you in the nets, {first_name}! "
                    f"Ask me anything about cricket: state/district trials & selections, seam/swing mechanics, match tactics, "
                    f"delivery stride biomechanics, or recovery diets. What are we sharpening today?"
                )
            else:
                fallback = (
                    f"Welcome to the nets {first_name} bhai! "
                    f"Cricket ke baare me kuch bhi pucho: local cricket trials, state/district selection process, bowling biomechanics, "
                    f"outswing/reverse-swing mechanics, pitch tactics, ya high-protein diet plan. Batao aaj kis topic pe baat karein?"
                )

        return {
            "reply": fallback,
            "language": lang,
            "has_context": bool(context_lines),
            "note": "coach_connected_response",
            "action_suggestions": action_suggestions
        }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"\n[PacePro AI] Starting Backend Server on http://localhost:{port} ...")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
