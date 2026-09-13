"""
PacePro AI - Fast Bowler Musculoskeletal & Kinetic Chain Weakness Engine
========================================================================
Sports-Science diagnostic engine calibrated against ECB, Cricket Australia, 
and Steffan Jones (Pacelab) biomechanical strength baselines.

Evaluates 5 critical fast-bowling physical links:
1. Lead-Leg Knee Brace & Quad Deceleration (VMO, Patellar Tendon)
2. Posterior Chain & Hip Extension Drive (Gluteus Max, Hamstrings)
3. Lumbar Shield & Core Anti-Rotation (Transverse Abdominis, Obliques, QL)
4. Scapular Deceleration & Shoulder Pull (Rotator Cuff, Lats, Posterior Deltoid)
5. Ankle Stiffness & Achilles Shock Absorption (Soleus, Tibialis Anterior)
"""

from typing import Dict, Any, List, Optional
import math


class StrengthDiagnosticsEngine:
    """
    Evaluates fast bowler physical test benchmarks and diagnoses specific
    muscular and kinetic chain vulnerabilities with customized corrective protocols.
    """

    def analyze_bowler_strength(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Takes bowler test inputs and computes:
        - Individual Muscle Chain Scores (0 - 100)
        - Primary Muscle Weakness Area
        - Biomechanical Breakdown on Bowling Performance
        - Injury Risk Advisory
        - Targeted Corrective Exercise Routine
        """
        weight_kg = float(data.get("weight_kg", 68.0))
        age = int(data.get("age", 20))
        track = data.get("strength_track", "gym_lifting")  # 'gym_lifting' | 'calisthenics_ground'
        bowling_arm = data.get("bowling_arm", "Right-Arm Fast")
        bowling_speed_kph = float(data.get("bowling_speed_kph", 132.0))

        # --- Test 1: Lead-Leg Knee Brace & Quad Deceleration ---
        # Benchmarks:
        # Gym: Bulgarian Split Squat working kg/hand >= 0.25x BW for 8 reps
        # Ground: Single-Leg Wall Sit >= 45s per leg OR Step-Down reps >= 15
        knee_score = 70.0
        knee_notes = []

        if track == "gym_lifting":
            split_squat_kg = float(data.get("split_squat_kg", 14.0))
            split_squat_reps = int(data.get("split_squat_reps", 8))
            ratio = split_squat_kg / max(35.0, weight_kg)
            ratio_score = min(100.0, (ratio / 0.25) * 85.0)
            rep_score = min(15.0, (split_squat_reps / 8.0) * 15.0)
            knee_score = round(ratio_score + rep_score, 1)
        else:
            wall_sit_sec = float(data.get("wall_sit_seconds", 35.0))
            knee_score = min(100.0, round((wall_sit_sec / 50.0) * 100.0, 1))

        wall_sit_reported = float(data.get("wall_sit_seconds", 40.0))
        if wall_sit_reported < 30.0:
            knee_score = min(knee_score, 58.0)
            knee_notes.append("Wall sit endurance under 30s: high risk of front knee buckling at FFC.")

        # --- Test 2: Posterior Chain & Hip Drive ---
        # Gym: Trap bar / Deadlift 1RM >= 1.8x BW (elite), 1.5x BW (good)
        # Ground: Max continuous Bodyweight Jump Squats (reps >= 35)
        hip_score = 70.0
        hip_notes = []

        if track == "gym_lifting" and data.get("does_deadlifts", True):
            deadlift_kg = float(data.get("deadlift_kg", 85.0))
            deadlift_reps = int(data.get("deadlift_reps", 5))
            est_1rm = deadlift_kg * (1.0 + deadlift_reps / 30.0)
            dl_ratio = est_1rm / max(35.0, weight_kg)
            hip_score = min(100.0, round((dl_ratio / 1.7) * 100.0, 1))
            if dl_ratio < 1.3:
                hip_notes.append(f"Posterior chain capacity ({dl_ratio:.2f}x BW) is below the 1.5x fast bowling baseline.")
        else:
            bw_squats = int(data.get("bodyweight_squats_reps", 25))
            hip_score = min(100.0, round((bw_squats / 40.0) * 100.0, 1))
            if bw_squats < 25:
                hip_notes.append("Lower body power output is low; inability to propel through run-up into crease.")

        # --- Test 3: Core Anti-Rotation & Lumbar Shield ---
        # Benchmark: Plank hold duration (ideal >= 90s) & Side Plank (ideal >= 60s/side)
        plank_sec = float(data.get("plank_seconds", 60.0))
        side_plank_sec = float(data.get("side_plank_seconds", 40.0))
        core_score = min(100.0, round(((plank_sec / 90.0) * 50.0) + ((side_plank_sec / 60.0) * 50.0), 1))
        core_notes = []

        if plank_sec < 45.0 or side_plank_sec < 30.0:
            core_notes.append("Oblique / transverse abdominis deficit leaves lumbar spine exposed to twisting shear.")

        # --- Test 4: Scapular Deceleration & Shoulder Pull ---
        # Benchmark: Strict Pull-ups >= 10 reps, Push-ups >= 30 reps.
        pullups = int(data.get("pullups_count", 6))
        pushups = int(data.get("pushups_reps", 25))
        upper_score = min(100.0, round(((pullups / 10.0) * 60.0) + ((pushups / 35.0) * 40.0), 1))
        upper_notes = []

        if pullups < 5:
            upper_notes.append("Latissimus dorsi & rotator cuff weakness impairs bowling arm pull-down velocity.")

        # --- Test 5: Ankle Stiffness & Achilles Shock Absorption ---
        # Benchmark: Single-leg calf raises >= 25 reps per leg
        calf_raises = int(data.get("calf_raises_reps", 18))
        ankle_score = min(100.0, round((calf_raises / 28.0) * 100.0, 1))
        ankle_notes = []

        if calf_raises < 18:
            ankle_notes.append("Soleus/Achilles spring deficit causes heel striking and shin splints under delivery impact.")

        # Check for user-reported stiffness/niggles
        reported_niggle = str(data.get("reported_niggle", "none")).lower()
        if "knee" in reported_niggle:
            knee_score = max(35.0, knee_score - 20.0)
            knee_notes.append("Active knee joint discomfort detected.")
        if "back" in reported_niggle or "lumbar" in reported_niggle:
            core_score = max(35.0, core_score - 25.0)
            core_notes.append("Active lumbar soreness detected: prioritize anti-rotation stabilization.")
        if "shin" in reported_niggle:
            ankle_score = max(35.0, ankle_score - 20.0)
            ankle_notes.append("Tibial/shin soreness reported: calf deceleration is failing.")
        if "shoulder" in reported_niggle:
            upper_score = max(35.0, upper_score - 20.0)
            upper_notes.append("Shoulder cuff strain reported: avoid heavy overhead loading.")

        # Compile Muscle Chain Health Dictionary
        chains = [
            {
                "id": "knee_brace",
                "name": "Lead-Leg Knee Brace & Quad Deceleration",
                "muscles": "Vastus Medialis (VMO), Rectus Femoris, Patellar Tendon",
                "score": round(knee_score, 1),
                "status": "Optimal" if knee_score >= 80 else "Moderate" if knee_score >= 60 else "Critical Weak Link",
                "bowling_function": "Absorbs 8-9x body weight at Front-Foot Contact (FFC) to prevent knee flexion collapse.",
                "notes": knee_notes or ["Eccentric quadriceps capacity supports solid knee brace rigidity."]
            },
            {
                "id": "core_lumbar",
                "name": "Lumbar Shield & Core Anti-Rotation",
                "muscles": "Transverse Abdominis, Internal/External Obliques, Quadratus Lumborum",
                "score": round(core_score, 1),
                "status": "Optimal" if core_score >= 80 else "Moderate" if core_score >= 60 else "Critical Weak Link",
                "bowling_function": "Braces spinal column against counter-rotation torsion and prevents lumbar stress fractures.",
                "notes": core_notes or ["Trunk rotational integrity protects facet joints during lateral flex."]
            },
            {
                "id": "posterior_chain",
                "name": "Posterior Chain & Hip Drive",
                "muscles": "Gluteus Maximus, Biceps Femoris (Hamstrings), Erector Spinae",
                "score": round(hip_score, 1),
                "status": "Optimal" if hip_score >= 80 else "Moderate" if hip_score >= 60 else "Critical Weak Link",
                "bowling_function": "Generates horizontal velocity from run-up into takeoff and anchors crease brake.",
                "notes": hip_notes or ["Hip extension capacity supplies foundational bowling stride propulsion."]
            },
            {
                "id": "shoulder_scapular",
                "name": "Scapular Pull-Through & Shoulder Armor",
                "muscles": "Latissimus Dorsi, Infraspinatus, Teres Minor, Posterior Deltoid",
                "score": round(upper_score, 1),
                "status": "Optimal" if upper_score >= 80 else "Moderate" if upper_score >= 60 else "Critical Weak Link",
                "bowling_function": "Pulls bowling arm through the release window and safely decelerates the joint post-release.",
                "notes": upper_notes or ["Rotator cuff endurance decelerates high arm rotational speed (over 2000 deg/s)."]
            },
            {
                "id": "ankle_achilles",
                "name": "Ankle Spring & Achilles Shock Absorption",
                "muscles": "Soleus, Gastrocnemius, Achilles Tendon, Tibialis Anterior",
                "score": round(ankle_score, 1),
                "status": "Optimal" if ankle_score >= 80 else "Moderate" if ankle_score >= 60 else "Critical Weak Link",
                "bowling_function": "Translates run-up momentum into vertical pop at BFC and cushions ground impact.",
                "notes": ankle_notes or ["Achilles tendon stiffness provides spring-like rebound off the turf."]
            }
        ]

        # Determine Weakest Link
        chains_sorted = sorted(chains, key=lambda x: x["score"])
        primary_weakness = chains_sorted[0]
        secondary_weakness = chains_sorted[1] if len(chains_sorted) > 1 else None

        overall_kinetic_index = round(sum(c["score"] for c in chains) / len(chains), 1)

        # Generate Tailored Corrective Routine
        corrective_routine = self._build_corrective_routine(primary_weakness["id"], track, bowling_arm)

        return {
            "kinetic_chain_score": overall_kinetic_index,
            "overall_rating": (
                "Elite Fast Bowling Durability" if overall_kinetic_index >= 85
                else "Competitive Pace Candidate" if overall_kinetic_index >= 70
                else "Vulnerable Kinetic Chain (Injury Risk Present)"
            ),
            "primary_weakness": {
                "id": primary_weakness["id"],
                "name": primary_weakness["name"],
                "score": primary_weakness["score"],
                "status": primary_weakness["status"],
                "affected_muscles": primary_weakness["muscles"],
                "bowling_impact": primary_weakness["bowling_function"],
                "clinical_notes": primary_weakness["notes"]
            },
            "secondary_weakness": {
                "id": secondary_weakness["id"],
                "name": secondary_weakness["name"],
                "score": secondary_weakness["score"],
                "affected_muscles": secondary_weakness["muscles"],
            } if secondary_weakness and secondary_weakness["score"] < 75 else None,
            "all_muscle_chains": chains,
            "injury_risk_advisory": self._build_injury_advisory(primary_weakness["id"]),
            "corrective_workout_routine": corrective_routine
        }

    def _build_injury_advisory(self, weakness_id: str) -> Dict[str, str]:
        advisories = {
            "knee_brace": {
                "risk_level": "High - Front Knee & Patella",
                "vulnerability": "Patellar Tendinopathy & Front Knee Flexion Collapse",
                "biomechanical_cause": "When VMO and quadriceps cannot withstand the 8x bodyweight front foot strike, the knee bends beyond 150°, leaking kinetic energy and dumping braking force into the lower back.",
                "priority_cue": "Maintain a straight, braced lead leg at release; do not allow knee to flex forward past your toes."
            },
            "core_lumbar": {
                "risk_level": "Severe - Lumbar Spine",
                "vulnerability": "L4/L5 Pars Interarticularis Bone Stress Fracture",
                "biomechanical_cause": "Weak obliques and transverse abdominis allow excessive counter-rotation and hyperextension during ball release, concentrating rotational shear onto lumbar vertebrae.",
                "priority_cue": "Lock your ribcage down to your pelvis; eliminate excessive side-bending at ball release."
            },
            "posterior_chain": {
                "risk_level": "Moderate - Hamstring & Sacroiliac Joint",
                "vulnerability": "Biceps Femoris Strain & Crease Deceleration Loss",
                "biomechanical_cause": "Hamstrings are forced to absorb entire hip deceleration because gluteus maximus is inactive, leading to late-spell sprint tears.",
                "priority_cue": "Squeeze glutes at takeoff; power the run-up through hip extension, not quad-pulling."
            },
            "shoulder_scapular": {
                "risk_level": "Moderate - Rotator Cuff & Labrum",
                "vulnerability": "Internal Impingement & Posterior Capsule Tightness",
                "biomechanical_cause": "Inadequate lat and rotator cuff strength fails to decelerate the bowling arm after ball release, causing the humerus to bang against the acromion.",
                "priority_cue": "Pull your bowling arm down across the opposite hip in follow-through rather than stopping it abruptly."
            },
            "ankle_achilles": {
                "risk_level": "High - Tibia & Achilles",
                "vulnerability": "Medial Tibial Stress Syndrome (Shin Splints) & Achilles Tendonitis",
                "biomechanical_cause": "Soleus fatigue prevents spring shock absorption, resulting in heavy heel striking that reverberates up the tibia during delivery stride.",
                "priority_cue": "Land on the mid-to-forefoot with a stiff ankle joint; avoid passive heel crashing."
            }
        }
        return advisories.get(weakness_id, advisories["knee_brace"])

    def _build_corrective_routine(self, weakness_id: str, track: str, bowling_arm: str) -> Dict[str, Any]:
        """
        Creates a periodized, targeted corrective routine aimed specifically at fixing
        the diagnosed weakness.
        """
        is_gym = (track == "gym_lifting")

        if weakness_id == "knee_brace":
            return {
                "protocol_name": "Lead-Leg Knee Plant Rigidity & Brace Protocol",
                "frequency": "3 sessions per week (Pre-bowling & S&C)",
                "target_focus": "Eliminate Knee Flexion Collapse at Front-Foot Contact",
                "phases": [
                    {
                        "phase_name": "Phase 1: Isometric Tendon Armor",
                        "duration": "10-12 Mins",
                        "exercises": [
                            {
                                "name": "Spanish Squat Iso-Hold" if is_gym else "Single-Leg Wall Sit (165° Lock)",
                                "sets": "3 sets",
                                "reps": "45-60 seconds hold",
                                "rest": "60s rest",
                                "coaching_cue": "Lock the lead knee at approximately 165°-170° angle. Feel intense tension through the VMO quad tendon without knee pain.",
                                "benefit": "Builds patellar tendon tensile stiffness to withstand 8x BW impact."
                            },
                            {
                                "name": "Eccentric Step-Downs (From 6-inch Box)",
                                "sets": "3 sets",
                                "reps": "8-10 reps per leg (4s lower)",
                                "rest": "45s rest",
                                "coaching_cue": "Lower heel slowly over 4 seconds, keeping lead knee perfectly tracking over middle toe without collapsing inward.",
                                "benefit": "Trains eccentric knee stability for front-foot landing."
                            }
                        ]
                    },
                    {
                        "phase_name": "Phase 2: Unilateral Brake Strength",
                        "duration": "15-20 Mins",
                        "exercises": [
                            {
                                "name": "Bulgarian Split Squats (Dumbbells)" if is_gym else "Rear-Foot Elevated Split Squat (Pause at Bottom)",
                                "sets": "4 sets",
                                "reps": "6-8 reps per leg",
                                "rest": "90s rest",
                                "coaching_cue": "Keep torso slightly forward like bowling approach; explode up driving through lead heel.",
                                "benefit": "Develops unilateral brake torque preventing knee collapse."
                            },
                            {
                                "name": "Single-Leg Lateral Stick & Land (Plyometric)",
                                "sets": "3 sets",
                                "reps": "5 reps per leg",
                                "rest": "60s rest",
                                "coaching_cue": "Bound laterally off back foot and land on lead foot, freezing like a statue for 2 seconds without wobbling.",
                                "benefit": "Replicates FFC braking forces and transfers ground reaction into ball velocity."
                            }
                        ]
                    }
                ]
            }

        elif weakness_id == "core_lumbar":
            return {
                "protocol_name": "Lumbar Armor & Anti-Rotation Core Shield",
                "frequency": "4 sessions per week (Daily core activation)",
                "target_focus": "Prevent Spinal Shear & Counter-Rotation Stress Fractures",
                "phases": [
                    {
                        "phase_name": "Phase 1: Anti-Extension & Rotational Bracing",
                        "duration": "12 Mins",
                        "exercises": [
                            {
                                "name": "Cable Pallof Press with Iso-Hold" if is_gym else "Resistance Band Anti-Rotation Press",
                                "sets": "3 sets",
                                "reps": "10 reps + 5s hold at extension",
                                "rest": "45s rest",
                                "coaching_cue": "Do not let the band twist your torso. Breathe into your diaphragm while keeping core as stiff as concrete.",
                                "benefit": "Strengthens internal/external obliques against fast bowler counter-rotation."
                            },
                            {
                                "name": "Copenhagen Adductor Plank",
                                "sets": "3 sets",
                                "reps": "30-45s per side",
                                "rest": "60s rest",
                                "coaching_cue": "Elevate top leg on a bench, keeping hips completely level with head and ankles.",
                                "benefit": "Locks pelvic-lumbar stability and protects the sacroiliac joint."
                            }
                        ]
                    },
                    {
                        "phase_name": "Phase 2: Kinetic Chain Bracing",
                        "duration": "15 Mins",
                        "exercises": [
                            {
                                "name": "Suitcase Farmer's Carry (Single Heavy DB)" if is_gym else "Single-Arm Overhead Kettlebell/Object Carry",
                                "sets": "3 sets",
                                "reps": "40 meters per side",
                                "rest": "60s rest",
                                "coaching_cue": "Walk in a straight line with upright posture. Do not lean towards or away from the weight.",
                                "benefit": "Anti-lateral flexion strength protects against bowling side-flexion injuries."
                            },
                            {
                                "name": "Deadbugs with Core Plate Press",
                                "sets": "3 sets",
                                "reps": "12 reps alternating",
                                "rest": "45s rest",
                                "coaching_cue": "Press lower back firmly into the floor; eliminate any arching in the lumbar spine.",
                                "benefit": "Ensures hip flexor movement without compromising lumbar neutrality."
                            }
                        ]
                    }
                ]
            }

        elif weakness_id == "posterior_chain":
            return {
                "protocol_name": "Posterior Chain & Glute Max Drive Protocol",
                "frequency": "3 sessions per week",
                "target_focus": "Maximize Takeoff Velocity & Crease Deceleration Power",
                "phases": [
                    {
                        "phase_name": "Phase 1: Hip Hinge & Posterior Armor",
                        "duration": "15 Mins",
                        "exercises": [
                            {
                                "name": "Trap Bar Deadlift (High Handles)" if is_gym else "Nordic Hamstring Falls (Bed or Partner Anchored)",
                                "sets": "4 sets",
                                "reps": "5 reps (Heavy)" if is_gym else "5 reps (Slow 4s eccentric)",
                                "rest": "90s rest",
                                "coaching_cue": "Drive floor away using glutes and hamstrings. Keep spine neutral and push hips through at lockout.",
                                "benefit": "Builds 1.5-1.8x BW posterior chain foundation essential for fast bowling."
                            },
                            {
                                "name": "Romanian Deadlift (RDL) with 3s Eccentric" if is_gym else "Single-Leg Glute Bridge with 3s Hold",
                                "sets": "3 sets",
                                "reps": "8-10 reps",
                                "rest": "60s rest",
                                "coaching_cue": "Push hips back towards the wall behind you until deep hamstring stretch is felt, then snap hips forward.",
                                "benefit": "Eliminates hamstring vulnerability during high-speed run-up approach."
                            }
                        ]
                    }
                ]
            }

        elif weakness_id == "shoulder_scapular":
            return {
                "protocol_name": "Rotator Cuff Deceleration & Lat Pull Protocol",
                "frequency": "3 sessions per week",
                "target_focus": "High-Speed Ball Release Pull-Through & Shoulder Armor",
                "phases": [
                    {
                        "phase_name": "Phase 1: Deceleration Cuff Stability",
                        "duration": "12 Mins",
                        "exercises": [
                            {
                                "name": "Cable / Band External Rotations at 90°",
                                "sets": "3 sets",
                                "reps": "12-15 reps per arm",
                                "rest": "45s rest",
                                "coaching_cue": "Keep elbow pinned at shoulder height; rotate forearm smoothly without arching spine.",
                                "benefit": "Strengthens infraspinatus to safely decelerate 2000°/s arm swing."
                            },
                            {
                                "name": "Weighted Pull-Ups or Lat Pulldowns" if is_gym else "Strict Pull-Ups (Chest-to-Bar)",
                                "sets": "4 sets",
                                "reps": "6-8 reps",
                                "rest": "75s rest",
                                "coaching_cue": "Initiate pull by retracting scapula down and back before elbow flexion.",
                                "benefit": "Lats provide the primary kinetic whip power during the ball delivery pull-down."
                            },
                            {
                                "name": "Prone Trap-3 / Y-T-W Raises",
                                "sets": "3 sets",
                                "reps": "10 reps per letter",
                                "rest": "45s rest",
                                "coaching_cue": "Squeeze mid-back blades together at top without shrugging traps up to ears.",
                                "benefit": "Improves scapular upward rotation and eliminates impingement."
                            }
                        ]
                    }
                ]
            }

        else:  # ankle_achilles
            return {
                "protocol_name": "Achilles Spring & Ankle Deceleration Protocol",
                "frequency": "3-4 sessions per week (Pre-bowling activation)",
                "target_focus": "Eliminate Shin Splints & Maximize Back-Foot Impact Spring",
                "phases": [
                    {
                        "phase_name": "Phase 1: Soleus & Achilles Isometric Stiffness",
                        "duration": "12 Mins",
                        "exercises": [
                            {
                                "name": "Bent-Knee Single-Leg Calf Raise (Wall Supported)",
                                "sets": "3 sets",
                                "reps": "15 reps + 5s hold at top",
                                "rest": "45s rest",
                                "coaching_cue": "Keep knee bent at 20-30° to target deep Soleus muscle rather than gastrocnemius.",
                                "benefit": "Soleus absorbs up to 6x BW at landing, directly preventing tibial stress fractures."
                            },
                            {
                                "name": "Tibialis Anterior Wall Raises",
                                "sets": "3 sets",
                                "reps": "20 reps continuous",
                                "rest": "45s rest",
                                "coaching_cue": "Lean back against wall, flex toes up towards shins as high as possible, hold for 1 second.",
                                "benefit": "Cures shin splints and improves foot clearance during delivery stride."
                            },
                            {
                                "name": "Ankle Pogo Hops (Stiff-Legged Springs)",
                                "sets": "3 sets",
                                "reps": "20 quick contacts",
                                "rest": "60s rest",
                                "coaching_cue": "Minimize ground contact time; bounce purely off the balls of your feet like a spring.",
                                "benefit": "Converts run-up horizontal speed into explosive vertical takeoff."
                            }
                        ]
                    }
                ]
            }


strength_engine = StrengthDiagnosticsEngine()
