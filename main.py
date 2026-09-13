"""
PacePro - Demo Entry Point

Usage examples:

    # Bowling analysis with all 3 views
    python main.py bowling --front front.mp4 --side1 side1.mp4 --side2 side2.mp4

    # Bowling analysis with just one view (missing views handled gracefully)
    python main.py bowling --side1 side1.mp4

    # Dietary targets
    python main.py dietary --age 22 --weight 75 --height 180 --sex male \\
        --activity high --goal performance

    # Workload from a JSON session-log file
    python main.py workload --sessions sessions.json

Add --no-llm to any bowling/dietary/workload command to skip LLM calls
(useful for a viva demo without needing network/API access -- all the
numeric measurements still run and print).
"""

import argparse
import json
import sys


def run_bowling(args):
    from pipeline import PaceProPipeline

    video_paths = {}
    if args.front:
        video_paths["front"] = args.front
    if args.side1:
        video_paths["side1"] = args.side1
    if args.side2:
        video_paths["side2"] = args.side2

    if not video_paths:
        print("Provide at least one of --front / --side1 / --side2", file=sys.stderr)
        sys.exit(1)

    pipeline = PaceProPipeline(pose_model_path=args.model_path)
    report = pipeline.analyze_bowling(video_paths, run_llm=not args.no_llm)

    print(json.dumps(report, indent=2, default=str))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nSaved report to {args.out}", file=sys.stderr)


def run_dietary(args):
    from planner_service import PlannerService

    ps = PlannerService()
    targets = ps.calculate_targets(
        age=args.age,
        weight_kg=args.weight,
        height_cm=args.height,
        sex=args.sex,
        activity_level=args.activity,
        goal=args.goal,
    )

    result = {"targets": targets}

    if not args.no_llm:
        result["meal_suggestions"] = ps.suggest_meals(targets)

    print(json.dumps(result, indent=2, default=str))


def run_workload(args):
    from planner_service import PlannerService

    with open(args.sessions) as f:
        sessions = json.load(f)

    ps = PlannerService()
    result = ps.calculate_workload(sessions)

    if result.get("success") and not args.no_llm:
        result["guidance_text"] = ps.generate_guidance_text(result)

    print(json.dumps(result, indent=2, default=str))


def build_parser():
    parser = argparse.ArgumentParser(description="PacePro AI - demo CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bowling_parser = subparsers.add_parser("bowling", help="Run bowling-video analysis")
    bowling_parser.add_argument("--front", help="Path to front-view video")
    bowling_parser.add_argument("--side1", help="Path to side1-view video")
    bowling_parser.add_argument("--side2", help="Path to side2-view video")
    bowling_parser.add_argument(
        "--model-path", default="models/pose_landmarker_full.task",
        help="Path to the mediapipe pose landmarker .task model"
    )
    bowling_parser.add_argument("--out", help="Optional path to save the JSON report")
    bowling_parser.add_argument("--no-llm", action="store_true")
    bowling_parser.set_defaults(func=run_bowling)

    dietary_parser = subparsers.add_parser("dietary", help="Calculate dietary targets")
    dietary_parser.add_argument("--age", type=int, required=True)
    dietary_parser.add_argument("--weight", type=float, required=True, help="kg")
    dietary_parser.add_argument("--height", type=float, required=True, help="cm")
    dietary_parser.add_argument("--sex", choices=["male", "female"], required=True)
    dietary_parser.add_argument(
        "--activity", choices=["sedentary", "light", "moderate", "high", "very_high"],
        required=True
    )
    dietary_parser.add_argument(
        "--goal", choices=["maintenance", "muscle_gain", "fat_loss", "performance"],
        required=True
    )
    dietary_parser.add_argument("--no-llm", action="store_true")
    dietary_parser.set_defaults(func=run_dietary)

    workload_parser = subparsers.add_parser("workload", help="Calculate training workload (ACWR)")
    workload_parser.add_argument(
        "--sessions", required=True,
        help="Path to a JSON file: a list of "
             '{"date": "YYYY-MM-DD", "duration_minutes": N, "intensity": "low|medium|high|max"}'
    )
    workload_parser.add_argument("--no-llm", action="store_true")
    workload_parser.set_defaults(func=run_workload)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
