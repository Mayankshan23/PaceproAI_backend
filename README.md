# PacePro AI - Backend

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...      # optional; pipeline degrades gracefully without it
```

You also need a MediaPipe pose landmarker model file (e.g.
`pose_landmarker_full.task` from Google's MediaPipe model zoo) at the
path you pass to `PoseAnalyzer` / `main.py --model-path`.

## Running

```bash
# Full bowling analysis (any subset of the 3 views is fine)
python main.py bowling --front front.mp4 --side1 side1.mp4 --side2 side2.mp4

# No LLM (pure CV numbers only -- useful for a viva without network access)
python main.py bowling --side1 side1.mp4 --no-llm

# Dietary targets
python main.py dietary --age 22 --weight 75 --height 180 --sex male --activity high --goal performance

# Workload (ACWR) from a session log JSON file
python main.py workload --sessions sessions.json
```

## No real bowling video / model file available?

`test_delivery_analyzer.py`, `test_fusion.py` and `test_pipeline.py`
build a fully deterministic **synthetic** pose sequence (no random
numbers -- a scripted run-up / plant / arm-swing motion) that matches
`PoseAnalyzer`'s exact output shape, and run it through the real
`DeliveryAnalyzer` -> `BiomechanicsCalculator` -> `ViewFusion` ->
`PaceProPipeline` chain. Useful to demonstrate the pipeline logic
live without needing a real video or the MediaPipe model file:

```bash
python test_pipeline.py
```

## Architecture

```
Video(s)
  -> PoseAnalyzer          (pose_detector.py)      [pre-existing]
  -> DeliveryAnalyzer      (delivery_analyzer.py)   [pre-existing, completed]
  -> BiomechanicsCalculator(biomechanics.py)        [new]
  -> ViewFusion            (view_fusion.py)         [new]
  -> BowlingLLMAnalyzer    (llm_analyzer.py)        [new]
  -> Final PacePro Report  (via pipeline.py)
```

`planner_service.py` (`PlannerService`) unifies and supersedes
`dietary_manager.py` and `workload_manager.py`, providing dietary targets,
cricket meal generation, session-RPE Acute:Chronic Workload Ratio (ACWR) tracking,
and combined student/full-time athlete fast-bowling protocols.

See the accompanying design write-up for the full list of what was
already implemented, what was missing, bugs found, and design
decisions made while building the biomechanics/fusion/LLM/dietary/
workload layers.

## Known limitations (stated openly, not hidden)

- Arm-slot classification and other 2D-landmark-derived angles are
  image-plane approximations, not true 3D biomechanics -- see
  comments in `biomechanics.py`.
- `DeliveryAnalyzer._calculate_foot_movement` compares consecutive
  entries in the pose_data list, not consecutive frame numbers. If
  pose detection drops a frame mid-sequence, that one movement value
  spans a slightly longer time gap than the rest. `analyze_view()`
  now surfaces this via `frame_gap_info` instead of hiding it, but
  the underlying detection method itself was intentionally left
  unchanged (see design write-up).
- ACWR is a training-load *monitoring* guideline, not a clinical or
  injury-prediction tool -- this is stated in the module and its
  output.
- Dietary targets use the Mifflin-St Jeor equation, a standard
  estimate -- not individualized medical/dietetic advice.
