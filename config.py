"""
PacePro - Central Configuration

All tunable constants for the NEW modules (biomechanics, view_fusion,
llm_analyzer, dietary_manager, workload_manager) live here so the
whole system has one place to adjust behaviour, and so numbers used
in a viva can be pointed to directly instead of hunted through code.

Note: PoseAnalyzer and DeliveryAnalyzer keep their OWN thresholds as
class constants (PLANT_THRESHOLD, MIN_VISIBILITY, etc.) inside their
own files. That is deliberate -- they are pre-existing, tested
modules and this file does not override their internal detection
methodology.
"""

import os
from pathlib import Path

# Load .env configuration
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path, override=False)
    else:
        load_dotenv(override=False)
except ImportError:
    pass

# ============================================================
# AI / LLM Configuration (OpenAI, Anthropic, Gemini)
# ============================================================

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-5",
    "gemini": "gemini-3.6-flash",
}

AI_PROVIDER = os.getenv("AI_PROVIDER", "openai").lower().strip()
AI_API_BASE_URL = os.getenv("AI_API_BASE_URL", "").strip() or None
AI_API_KEY = (
    os.getenv("AI_API_KEY", "").strip()
    or os.getenv("OPENAI_API_KEY", "").strip()
    or os.getenv("ANTHROPIC_API_KEY", "").strip()
    or os.getenv("GEMINI_API_KEY", "").strip()
    or os.getenv("LLM_API_KEY", "").strip()
)

API_MODEL_NAME = (
    os.getenv("API_MODEL_NAME", "").strip()
    or os.getenv("AI_MODEL_NAME", "").strip()
    or DEFAULT_MODELS.get(AI_PROVIDER, "gpt-4o-mini")
)

LLM_MODEL = API_MODEL_NAME
LLM_MAX_TOKENS = 4500
LLM_TEMPERATURE = 0.4  # low temperature: analysis, not creative writing

# Backward compatibility alias
ANTHROPIC_API_KEY_ENV_VAR = "AI_API_KEY"


def get_ai_provider() -> str:
    """Returns active AI provider: 'openai', 'anthropic', or 'gemini'."""
    val = os.getenv("AI_PROVIDER", "").lower().strip()
    return val if val else AI_PROVIDER


def get_ai_api_key() -> str:
    """Retrieves AI API key from AI_API_KEY or provider-specific variables."""
    return (
        os.getenv("AI_API_KEY", "").strip()
        or os.getenv("OPENAI_API_KEY", "").strip()
        or os.getenv("ANTHROPIC_API_KEY", "").strip()
        or os.getenv("GEMINI_API_KEY", "").strip()
        or os.getenv("LLM_API_KEY", "").strip()
        or AI_API_KEY
    )


def get_ai_base_url() -> str | None:
    """Optional base URL for OpenAI-compatible endpoints (Ollama, LM Studio, Groq, etc.)."""
    val = os.getenv("AI_API_BASE_URL", "").strip()
    return val if val else AI_API_BASE_URL


def get_ai_model_name() -> str:
    """Returns model name from API_MODEL_NAME, AI_MODEL_NAME, or provider default."""
    provider = get_ai_provider()
    return (
        os.getenv("API_MODEL_NAME", "").strip()
        or os.getenv("AI_MODEL_NAME", "").strip()
        or API_MODEL_NAME
        or DEFAULT_MODELS.get(provider, "gpt-4o-mini")
    )


def is_llm_configured() -> bool:
    """Returns True if primary or any fallback provider has a key/base_url."""
    return len(get_all_llm_configs()) > 0


def get_ai_config() -> dict:
    """Returns primary AI configuration dictionary."""
    return {
        "provider": get_ai_provider(),
        "api_key": get_ai_api_key(),
        "base_url": get_ai_base_url(),
        "model": get_ai_model_name(),
        "role": "primary",
    }


def get_fallback_configs() -> list[dict]:
    """Returns list of configured fallback AI providers."""
    fallbacks = []

    # Fallback 1
    fb1_provider = (
        os.getenv("FALLBACK_AI_PROVIDER", "").strip()
        or os.getenv("AI_FALLBACK_PROVIDER", "").strip()
    )
    fb1_key = (
        os.getenv("FALLBACK_AI_API_KEY", "").strip()
        or os.getenv("AI_FALLBACK_API_KEY", "").strip()
    )
    fb1_url = (
        os.getenv("FALLBACK_AI_API_BASE_URL", "").strip()
        or os.getenv("AI_FALLBACK_API_BASE_URL", "").strip()
        or None
    )
    fb1_model = (
        os.getenv("FALLBACK_API_MODEL_NAME", "").strip()
        or os.getenv("FALLBACK_AI_MODEL_NAME", "").strip()
        or os.getenv("AI_FALLBACK_MODEL_NAME", "").strip()
    )

    if fb1_key or fb1_url:
        prov = fb1_provider.lower() if fb1_provider else "gemini"
        model = fb1_model if fb1_model else DEFAULT_MODELS.get(prov, "gemini-3.6-flash")
        fallbacks.append({
            "provider": prov,
            "api_key": fb1_key,
            "base_url": fb1_url,
            "model": model,
            "role": "fallback_1",
        })

    # Fallback 2 (Optional)
    fb2_provider = os.getenv("FALLBACK_2_AI_PROVIDER", "").strip()
    fb2_key = os.getenv("FALLBACK_2_AI_API_KEY", "").strip()
    fb2_url = os.getenv("FALLBACK_2_AI_API_BASE_URL", "").strip() or None
    fb2_model = os.getenv("FALLBACK_2_API_MODEL_NAME", "").strip()

    if fb2_key or fb2_url:
        prov = fb2_provider.lower() if fb2_provider else "anthropic"
        model = fb2_model if fb2_model else DEFAULT_MODELS.get(prov, "claude-sonnet-5")
        fallbacks.append({
            "provider": prov,
            "api_key": fb2_key,
            "base_url": fb2_url,
            "model": model,
            "role": "fallback_2",
        })

    return fallbacks


def get_all_llm_configs() -> list[dict]:
    """Returns primary config followed by all active fallback configs."""
    configs = []
    primary = get_ai_config()
    if primary["api_key"] or primary["base_url"]:
        configs.append(primary)
    configs.extend(get_fallback_configs())
    return configs


def get_anthropic_api_key() -> str:
    """Legacy helper for Anthropic-specific callers."""
    return get_ai_api_key()


# ============================================================
# BIOMECHANICS -- arm slot classification thresholds
#
# Angle of the (shoulder -> wrist) vector of the bowling arm,
# measured from horizontal, at the release frame.
# These bands are a simplified teaching heuristic (not a
# validated biomechanics standard) -- appropriate for a
# prototype/college project, clearly explainable in a viva.
# ============================================================

ARM_SLOT_BANDS = [
    # (min_angle_deg, max_angle_deg, label)
    (60, 90, "high_arm"),        # classical / round-arm
    (30, 60, "three_quarter"),
    (0, 30, "side_arm"),         # sling / round-arm low
]

# Arm slot depends on seeing the vertical plane of the arm swing
# cleanly -- most reliable from the front-on camera.
ARM_SLOT_PREFERRED_VIEW = "front"


# ============================================================
# VIEW FUSION
# ============================================================

# For each derived measurement, which view(s) are biomechanically
# most trustworthy, in preference order. Side-on views avoid the
# foreshortening that a front-on camera introduces for angles that
# move mostly in the sagittal (side) plane -- e.g. knee flexion.
# Arm slot is the opposite case (see ARM_SLOT_PREFERRED_VIEW above).
MEASUREMENT_VIEW_PREFERENCE = {
    "back_knee_flexion_at_bfc": ["side1", "side2", "front"],
    "front_knee_angle_at_ffc": ["side1", "side2", "front"],
    "front_knee_brace_change": ["side1", "side2", "front"],
    "release_elbow_extension": ["side1", "side2", "front"],
    "trunk_lean_at_release": ["front", "side1", "side2"],
    "hip_shoulder_separation": ["side1", "side2", "front"],
    "arm_slot": ["front", "side1", "side2"],
}

# If two views report the same angle within this tolerance, treat
# them as agreeing and average them instead of picking one.
FUSION_AGREEMENT_TOLERANCE_DEG = 8.0


# ============================================================
# DIETARY MANAGEMENT
# ============================================================

ACTIVITY_MULTIPLIERS = {
    "sedentary": 1.2,
    "light": 1.375,      # light training, 1-3 sessions/week
    "moderate": 1.55,    # regular training, 3-5 sessions/week
    "high": 1.725,       # daily training / match season
    "very_high": 1.9,    # multiple sessions/day, heavy match load
}

GOAL_CALORIE_ADJUSTMENT = {
    "maintenance": 0.0,
    "muscle_gain": 0.12,     # +12% surplus
    "fat_loss": -0.18,       # -18% deficit
    "performance": 0.0,      # maintenance calories, carb-weighted macros
}

# grams of protein per kg bodyweight -- athlete range, not general population
PROTEIN_G_PER_KG = {
    "maintenance": 1.8,
    "muscle_gain": 2.0,
    "fat_loss": 2.2,      # higher protein helps preserve muscle in a deficit
    "performance": 1.8,
}

FAT_PERCENT_OF_CALORIES = 0.25  # remainder after protein goes to carbs


# ============================================================
# WORKLOAD MANAGEMENT (WRC)
#
# Uses the session-RPE method (Foster et al.) to get a training
# load number, and the Acute:Chronic Workload Ratio (ACWR) concept
# widely taught in sports science / cricket fast-bowling workload
# literature. This is a TRANSPARENT, EXPLAINABLE metric -- not a
# medical or injury-prediction tool, and the module says so.
# ============================================================

# session load = duration_minutes * RPE_FACTOR[intensity]
INTENSITY_RPE_FACTOR = {
    "low": 3,
    "medium": 5,
    "high": 7,
    "max": 9,
}

ACUTE_WINDOW_DAYS = 7
CHRONIC_WINDOW_DAYS = 28

# Commonly cited "sweet spot" band for ACWR in sports-science literature.
ACWR_LOW = 0.8
ACWR_HIGH = 1.3
ACWR_SPIKE = 1.5  # above this: flag a load spike


# ============================================================
# MONGODB ATLAS DATABASE CONFIGURATION
# ============================================================
MONGODB_URI = os.getenv("MONGODB_URI", "").strip()
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "pacepro_db").strip()

def get_mongodb_uri() -> str:
    return os.getenv("MONGODB_URI", "").strip() or MONGODB_URI

def get_mongodb_db_name() -> str:
    return os.getenv("MONGODB_DB_NAME", "pacepro_db").strip() or MONGODB_DB_NAME
