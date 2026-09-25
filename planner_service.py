"""
PacePro AI - Unified Planner Service (Nutrition, Workload & Combined Protocol)
==============================================================================
Consolidates and supersedes:
  - dietary_manager.py (Mifflin-St Jeor targets, macro distribution, athletic meal plans)
  - workload_manager.py (Session-RPE, ACWR acute:chronic workload ratios, trend, zones)

Capabilities:
1. Dietary Management:
   - Calculate daily calorie & macro targets (protein, fat, carbs) via Mifflin-St Jeor
   - Generate cricket-specific daily meal breakdowns adapted to food preferences and budgets
   - LLM-assisted cricket meal suggestions
2. Workload Management (WRC):
   - Acute:Chronic Workload Ratio (ACWR) tracking using session-RPE (Foster et al.)
   - 7-day acute and 28-day chronic daily load monitoring
   - Workload spike alerts, ECB bowling directives compliance, and injury-risk advisories
3. Combined Athletic Planning:
   - 7-day synchronized fast bowling workload & nutrition protocol
   - Student vs Full-Time Athlete schedule and timetable adaptation
   - Single-call LLM workout & diet plan generation with graceful fallbacks
"""

import json
import logging
import os
import re
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, Optional, List

logger = logging.getLogger("pacepro.planner")

import config

# Optional LLM integration (graceful fallback if uninstalled or offline)
try:
    from llm_analyzer import LLMClient, LLMUnavailableError
except ImportError:
    class LLMUnavailableError(Exception):
        pass

    class LLMClient:
        def generate_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
            raise LLMUnavailableError("LLM client not configured or dependencies not installed.")

        def generate_text(self, system_prompt: str, user_prompt: str) -> str:
            raise LLMUnavailableError("LLM client not configured or dependencies not installed.")


# =====================================================================
# CONSTANTS & LOOKUPS
# =====================================================================

ACTIVITY_MULTIPLIERS = {
    "sedentary": 1.2,
    "light": 1.375,
    "moderate": 1.55,
    "high": 1.725,
    "very_active": 1.725,
    "very_high": 1.9,
    "extra_active": 1.9,
}

VALID_SEXES = ["male", "female"]
VALID_ACTIVITY_LEVELS = list(ACTIVITY_MULTIPLIERS.keys())


class PlannerService:
    """
    Unified sports-science planning engine for PacePro AI.
    Replaces and merges DietaryManager and WorkloadManager.
    """

    # -----------------------------------------------------------------
    # 1. DIETARY MANAGEMENT
    # -----------------------------------------------------------------

    def calculate_targets(
        self,
        age: int,
        weight_kg: float,
        height_cm: float,
        sex: str,
        activity_level: str,
        goal: str,
    ) -> Dict[str, Any]:
        """
        Calculates daily calorie and macronutrient targets using the
        Mifflin-St Jeor equation and sports-nutrition guidelines.
        """
        self._validate_dietary_inputs(age, weight_kg, height_cm, sex, activity_level, goal)

        bmr = self.calculate_bmr(weight_kg, height_cm, age, sex)

        # Activity multiplier lookup (handles both config keys and aliases)
        act_key = activity_level.lower().strip()
        activity_multiplier = ACTIVITY_MULTIPLIERS.get(
            act_key, config.ACTIVITY_MULTIPLIERS.get(act_key, 1.55)
        )
        tdee = bmr * activity_multiplier

        # Calorie adjustment lookup
        goal_key = goal.lower().strip()
        calorie_adjustment = config.GOAL_CALORIE_ADJUSTMENT.get(
            goal_key,
            0.12 if "muscle" in goal_key
            else -0.18 if "fat" in goal_key or "cut" in goal_key
            else 0.05 if "speed" in goal_key
            else 0.0
        )
        target_calories = tdee * (1 + calorie_adjustment)

        # Protein per kg bodyweight
        protein_factor = config.PROTEIN_G_PER_KG.get(
            goal_key,
            2.2 if "fat" in goal_key
            else 2.0 if "muscle" in goal_key
            else 1.8
        )
        protein_g = protein_factor * weight_kg
        protein_calories = protein_g * 4

        fat_calories = target_calories * config.FAT_PERCENT_OF_CALORIES
        fat_g = fat_calories / 9

        carb_calories = target_calories - protein_calories - fat_calories
        carb_g = max(carb_calories, 0) / 4  # floor at 0

        return {
            "inputs": {
                "age": age,
                "weight_kg": weight_kg,
                "height_cm": height_cm,
                "sex": sex,
                "activity_level": activity_level,
                "goal": goal,
            },
            "bmr_kcal": round(bmr),
            "tdee_kcal": round(tdee),
            "target_calories_kcal": round(target_calories),
            "macros": {
                "protein_g": round(protein_g),
                "fat_g": round(fat_g),
                "carbs_g": round(carb_g),
            },
            "method_note": (
                "BMR via Mifflin-St Jeor equation; TDEE = BMR x activity "
                "multiplier; goal adjustment applied to TDEE; protein set "
                "per kg bodyweight for athletes, fat as a fixed % of "
                "calories, carbs fill the remainder. This is a prototype "
                "estimate, not individualized medical/dietetic advice."
            ),
        }

    def suggest_meals(self, targets: Dict[str, Any], llm_client: Optional[Any] = None) -> Dict[str, Any]:
        """
        Turns calculated targets into cricket-oriented meal suggestions using LLM
        or a fallback structured breakdown.
        """
        client = llm_client or LLMClient()

        system_prompt = (
            "You are a sports-nutrition assistant for a cricket fast "
            "bowler. You will be given ALREADY-CALCULATED daily calorie "
            "and macro targets. Do not change, recalculate, or second-"
            "guess these numbers -- treat them as fixed facts. Your job "
            "is only to suggest a practical day's worth of meals "
            "(breakfast, lunch, dinner, 1-2 snacks) that roughly fits "
            "those targets, with common foods relevant to Indian/"
            "international cricket players. Respond ONLY with a JSON "
            "object of this shape, no markdown fences:\n"
            "{\n"
            '  "meals": [{"name": "Breakfast", "suggestion": "..."}, ...],\n'
            '  "notes": "1-2 sentences of practical guidance"\n'
            "}"
        )

        user_prompt = (
            f"Daily target: {targets.get('target_calories_kcal', 2500)} kcal, "
            f"protein {targets.get('macros', {}).get('protein_g', 140)}g, "
            f"fat {targets.get('macros', {}).get('fat_g', 70)}g, "
            f"carbs {targets.get('macros', {}).get('carbs_g', 300)}g. "
            f"Goal: {targets.get('inputs', {}).get('goal', 'performance')}."
        )

        try:
            return client.generate_json(system_prompt, user_prompt)
        except Exception as exc:
            # Deterministic fallback meal suggestions for viva/offline demo
            return {
                "meals": [
                    {
                        "name": "Breakfast",
                        "suggestion": "Rolled oats with whey/plant protein, sliced banana, and 1 tbsp peanut butter."
                    },
                    {
                        "name": "Mid-Morning / Pre-Spell",
                        "suggestion": "Multigrain toast with honey, soaked almonds, dates, and coconut water."
                    },
                    {
                        "name": "Lunch",
                        "suggestion": "Grilled chicken breast or paneer/soya curry, brown basmati rice, steamed greens, and Greek curd."
                    },
                    {
                        "name": "Evening Snack",
                        "suggestion": "Roasted chana, mixed pumpkin & flax seeds, green tea."
                    },
                    {
                        "name": "Dinner",
                        "suggestion": "3 multigrain rotis, yellow moong dal with paneer or grilled fish, spinach salad, and warm turmeric milk."
                    }
                ],
                "notes": (
                    "Structured cricket-athletic meal plan calculated for high glycogen replenishment, "
                    "tissue repair, and lumbar spinal health."
                ),
                "_status": "offline_fallback" if isinstance(exc, LLMUnavailableError) else "fallback",
            }

    def generate_diet_plan(self, req_data: Dict[str, Any], llm_client: Optional[Any] = None) -> Dict[str, Any]:
        """
        Full daily meal plan generator matching the /api/generate-diet format.
        Uses exact sports-science calculation (Mifflin-St Jeor) and calls LLM
        to generate tailored meal suggestions with dynamic fallbacks.
        """
        age = int(req_data.get("age", 20))
        weight = float(req_data.get("weight", req_data.get("weight_kg", 70.0)))
        height = float(req_data.get("height", req_data.get("height_cm", 175.0)))
        sex = req_data.get("sex", "male").lower()
        activity = req_data.get("activity", req_data.get("activity_level", "high")).lower()
        goal = req_data.get("goal", "performance").lower()
        food_pref = (req_data.get("foodPref") or req_data.get("food_preference") or "non-veg").lower()
        budget_weekly = float(req_data.get("budgetWeekly", req_data.get("budget", {}).get("amount", 3500.0)))
        allergies = req_data.get("allergies", [])
        if not isinstance(allergies, list):
            allergies = [str(allergies)]
        avoid_foods = req_data.get("avoidFoods", req_data.get("avoid_foods", []))
        if not isinstance(avoid_foods, list):
            avoid_foods = [str(avoid_foods)]
        meal_freq = int(req_data.get("mealFreq", req_data.get("meal_frequency", 5)))
        cooking_pref = str(req_data.get("cookingPref", req_data.get("cooking_preference", "home")))
        recovery_note = str(req_data.get("recoveryNote", req_data.get("recovery_note", "none")))

        # Validate & normalize
        if activity not in ACTIVITY_MULTIPLIERS and activity not in config.ACTIVITY_MULTIPLIERS:
            activity = "high"
        if goal not in config.GOAL_CALORIE_ADJUSTMENT:
            goal = "performance"

        targets = self.calculate_targets(
            age=age,
            weight_kg=weight,
            height_cm=height,
            sex=sex,
            activity_level=activity,
            goal=goal,
        )

        total_cals = targets["target_calories_kcal"]
        protein_g = targets["macros"]["protein_g"]
        fat_g = targets["macros"]["fat_g"]
        carbs_g = targets["macros"]["carbs_g"]

        profile = {
            "age": age,
            "weight": weight,
            "height": height,
            "sex": sex,
            "goal": goal,
            "food_pref": food_pref,
            "budget_weekly": budget_weekly,
            "allergies": allergies,
            "avoid_foods": avoid_foods,
            "meal_freq": meal_freq,
            "cooking_pref": cooking_pref,
            "recovery_note": recovery_note,
        }

        # Concurrently generate meals and AI coaching assessment to reduce generation time by 50%
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            meals_future = executor.submit(self._generate_llm_meals, targets, profile, llm_client)
            ai_future = executor.submit(
                self.generate_ai_coaching_assessment,
                athlete_profile={
                    "age": age,
                    "weight": weight,
                    "height": height,
                    "sex": sex,
                    "goal": goal,
                    "foodPref": food_pref
                },
                nutrition_metrics={
                    "bmr": self.calculate_bmr(weight, height, age, sex),
                    "tdee": round(total_cals / (1.15 if goal == "performance" else 1.0)),
                    "target_calories": total_cals,
                    "protein_g": protein_g,
                    "carbs_g": carbs_g,
                    "fats_g": fat_g
                },
                workload_metrics={
                    "acwr_score": 1.18,
                    "acwr_status": "Optimal Range (1.10 - 1.30 ACWR)",
                    "total_weekly_overs": 16,
                    "gym_days": 3,
                    "ground_days": 4,
                    "max_overs_days": 2
                },
                llm_client=llm_client
            )
            meals = meals_future.result()
            ai_assessment = ai_future.result()

        daily_budget = sum(m.get("estimated_cost_inr", 0) for m in meals)
        if daily_budget <= 0:
            daily_budget = round(budget_weekly / 7)

        return {
            "daily_targets": {
                "calories": total_cals,
                "protein_g": protein_g,
                "carbs_g": carbs_g,
                "fats_g": fat_g,
            },
            "estimated_budget": {
                "daily_inr": daily_budget,
                "weekly_inr": round(daily_budget * 7),
            },
            "meals": meals,
            "ai_coaching_assessment": ai_assessment,
            "ai_assessment": ai_assessment,
        }

    def _generate_llm_meals(
        self,
        targets: Dict[str, Any],
        profile: Dict[str, Any],
        llm_client: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Calls LLM to generate cricket-specific athletic meal breakdown tailored
        to the exact pre-calculated sports-science calorie/macro figures,
        food preference, allergies, budget, and meal frequency.
        """
        client = llm_client or LLMClient()
        total_cals = targets.get("target_calories_kcal", 2800)
        protein_g = targets.get("macros", {}).get("protein_g", 150)
        fat_g = targets.get("macros", {}).get("fat_g", 65)
        carbs_g = targets.get("macros", {}).get("carbs_g", 380)

        age = profile.get("age", 20)
        weight = profile.get("weight", 70.0)
        height = profile.get("height", 175.0)
        sex = profile.get("sex", "male")
        goal = profile.get("goal", "performance")
        food_pref = profile.get("food_pref", "non-veg")
        allergies = profile.get("allergies", [])
        avoid_foods = profile.get("avoid_foods", [])
        meal_freq = max(3, min(6, int(profile.get("meal_freq", 5))))
        budget_weekly = float(profile.get("budget_weekly", 3500.0))
        daily_budget = round(budget_weekly / 7)

        system_prompt = (
            "You are PacePro AI's Lead Sports Nutritionist and Fast Bowling Performance Director.\n"
            "You are given PRE-CALCULATED, verified sports-science daily targets for a cricket fast bowler:\n"
            f"- Daily Calories: {total_cals} kcal\n"
            f"- Protein: {protein_g}g\n"
            f"- Carbohydrates: {carbs_g}g\n"
            f"- Fats: {fat_g}g\n"
            f"- Number of Meals: Exactly {meal_freq} spaced meals\n"
            f"- Daily Target Budget: ~₹{daily_budget} INR\n\n"
            "CRITICAL RULES:\n"
            "1. DO NOT change or recalculate these total numbers. Distribute these exact totals across the meals.\n"
            f"2. Strict Food Preference: '{food_pref}'. If vegetarian/vegan/eggetarian, NEVER include forbidden meats.\n"
            f"3. Strict Allergy & Avoidance Constraints: NEVER include allergens: {json.dumps(allergies)}, and avoid: {json.dumps(avoid_foods)}.\n"
            "4. Provide realistic, performance-oriented foods for cricket bowlers (e.g. Rolled Oats, Sattu, Paneer, Curd, Moong Dal, Sweet Potatoes, Eggs/Chicken/Fish if non-veg).\n"
            "5. Respond ONLY with a valid JSON array of meal objects, no markdown fences:\n"
            "[\n"
            "  {\n"
            '    "type": "Breakfast",\n'
            '    "time": "07:30 AM",\n'
            '    "title": "Fast Bowler High-Protein Energy Bowl",\n'
            '    "items": ["Paneer Bhurji / Boiled Eggs (150g)", "100g Rolled Oats with Whey / Plant Isolate", "1 Banana with 1 tbsp Peanut Butter"],\n'
            '    "calories": 650,\n'
            '    "protein_g": 38,\n'
            '    "carbs_g": 80,\n'
            '    "fats_g": 18,\n'
            '    "estimated_cost_inr": 110\n'
            "  }\n"
            "]"
        )

        user_prompt = (
            f"Generate exactly {meal_freq} athletic meals for a {age}yo {sex} fast bowler ({weight}kg, {height}cm).\n"
            f"Goal: {goal.replace('_', ' ').title()} | Diet: {food_pref.title()}\n"
            f"Allergies to EXCLUDE: {', '.join(allergies) if allergies else 'None'}\n"
            f"Calculated Targets: {total_cals} kcal | {protein_g}g Protein | {carbs_g}g Carbs | {fat_g}g Fats | Budget ~₹{daily_budget}/day."
        )

        try:
            parsed = client.generate_json(system_prompt, user_prompt)
            if isinstance(parsed, dict) and "meals" in parsed:
                parsed = parsed["meals"]
            if isinstance(parsed, list) and len(parsed) >= 2:
                valid_meals = []
                for m in parsed:
                    if isinstance(m, dict) and ("type" in m or "title" in m):
                        raw_items = m.get("items", [])
                        if isinstance(raw_items, str):
                            items_list = [raw_items]
                        elif isinstance(raw_items, list):
                            items_list = [str(it) for it in raw_items]
                        else:
                            items_list = ["Wholesome athletic meal ingredients"]

                        valid_meals.append({
                            "type": str(m.get("type", "Meal")),
                            "time": str(m.get("time", "08:00 AM")),
                            "title": str(m.get("title", "Athletic Performance Meal")),
                            "items": items_list,
                            "calories": int(m.get("calories", round(total_cals / len(parsed)))),
                            "protein_g": int(m.get("protein_g", round(protein_g / len(parsed)))),
                            "carbs_g": int(m.get("carbs_g", round(carbs_g / len(parsed)))),
                            "fats_g": int(m.get("fats_g", round(fat_g / len(parsed)))),
                            "estimated_cost_inr": int(m.get("estimated_cost_inr", round(daily_budget / len(parsed)))),
                        })
                if len(valid_meals) >= 2:
                    return valid_meals
        except Exception as e:
            logger.warning(f"[PlannerService] LLM meal generation failed or offline: {e}")

        # Fallback to calculated meal builder
        return self._build_fallback_meals(targets, profile)

    def _build_fallback_meals(self, targets: Dict[str, Any], profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Deterministic, sports-science calculated meal generator that dynamically
        adapts to meal frequency, food preference, allergies, and calorie/macro totals.
        """
        total_cals = targets.get("target_calories_kcal", 2800)
        protein_g = targets.get("macros", {}).get("protein_g", 150)
        fat_g = targets.get("macros", {}).get("fat_g", 65)
        carbs_g = targets.get("macros", {}).get("carbs_g", 380)

        food_pref = profile.get("food_pref", "non-veg").lower()
        is_veg = "veg" in food_pref and "non" not in food_pref
        is_vegan = "vegan" in food_pref
        is_egg = "egg" in food_pref

        allergies = [str(a).lower() for a in profile.get("allergies", [])]
        has_dairy_allergy = any("dairy" in a or "milk" in a or "lactose" in a for a in allergies)
        has_nut_allergy = any("nut" in a or "peanut" in a for a in allergies)
        has_gluten_allergy = any("gluten" in a or "wheat" in a for a in allergies)
        has_soy_allergy = any("soy" in a for a in allergies)
        has_egg_allergy = any("egg" in a for a in allergies)

        meal_freq = max(3, min(6, int(profile.get("meal_freq", 5))))
        budget_weekly = float(profile.get("budget_weekly", 3500.0))
        daily_budget = round(budget_weekly / 7)

        # Protein item builder
        def get_breakfast_protein():
            if is_vegan:
                return "Tofu Scramble (150g) with turmeric & nutritional yeast" if not has_soy_allergy else "Sprouted Green Moong & Besan Chilla (3 chillas)"
            if is_veg:
                return "Paneer Bhurji (150g)" if not has_dairy_allergy else "Sprouted Moong Chilla with hemp seeds"
            if is_egg or not has_egg_allergy:
                return "4 Boiled Egg Whites + 1 Whole Egg Omelette"
            return "Paneer Scramble or Soya Bhurji (150g)"

        def get_lunch_protein():
            if is_vegan:
                return "200g Soya Chunks Curry with Spinach" if not has_soy_allergy else "200g Chickpea (Chole) & Dal Curry"
            if is_veg:
                return "200g Low-Fat Paneer / Soya Chunks Curry" if not has_dairy_allergy else "200g Soya & Black Chana Curry"
            return "200g Grilled Chicken Breast or Fish Fillet"

        def get_dinner_protein():
            if is_vegan:
                return "Yellow Moong Dal Tadka with Steamed Tofu" if not has_soy_allergy else "Thick Rajma / Dal Tadka with roasted seeds"
            if is_veg:
                return "Thick Moong Dal with Sautéed Paneer (120g)" if not has_dairy_allergy else "Moong Dal with Soya Paneer"
            return "150g Grilled Fish or Skinless Chicken Curry with Dal"

        def get_carb_base():
            if has_gluten_allergy:
                return "Steamed Brown Basmati Rice (200g) / Sweet Potato"
            return "3 Whole Wheat Multigrain Rotis or Brown Basmati Rice"

        def get_nuts_item():
            if has_nut_allergy:
                return "Roasted Pumpkin & Sunflower Seeds (30g)"
            return "12 Soaked Almonds & 3 Royal Dates"

        if meal_freq == 3:
            # 3 Large Meals Split
            return [
                {
                    "type": "Breakfast",
                    "time": "08:00 AM",
                    "title": "Power Athletic Breakfast",
                    "items": [
                        get_breakfast_protein(),
                        "100g Rolled Oats with Plant/Whey Protein" if not has_gluten_allergy else "Brown Rice Flakes (Poha) with Sprouts",
                        get_nuts_item(),
                        "1 Ripe Banana with Tender Coconut Water"
                    ],
                    "calories": round(total_cals * 0.35),
                    "protein_g": round(protein_g * 0.35),
                    "carbs_g": round(carbs_g * 0.35),
                    "fats_g": round(fat_g * 0.35),
                    "estimated_cost_inr": round(daily_budget * 0.35),
                },
                {
                    "type": "Lunch",
                    "time": "01:30 PM",
                    "title": "Performance Fuel & Repair Lunch",
                    "items": [
                        get_lunch_protein(),
                        get_carb_base(),
                        "Mixed Steamed Veggies (Broccoli, Carrots)",
                        "1 Cup Greek Curd / Probiotic Dahi" if not has_dairy_allergy and not is_vegan else "Sprout Salad with Lemon Dressing"
                    ],
                    "calories": round(total_cals * 0.40),
                    "protein_g": round(protein_g * 0.40),
                    "carbs_g": round(carbs_g * 0.40),
                    "fats_g": round(fat_g * 0.35),
                    "estimated_cost_inr": round(daily_budget * 0.40),
                },
                {
                    "type": "Dinner",
                    "time": "08:30 PM",
                    "title": "Overnight Tissue Repair & Lumbar Recovery",
                    "items": [
                        get_dinner_protein(),
                        get_carb_base(),
                        "Green Spinach, Cucumber & Beetroot Salad",
                        "1 Glass Warm Golden Turmeric Milk" if not has_dairy_allergy and not is_vegan else "Warm Golden Almond/Oat Milk"
                    ],
                    "calories": round(total_cals * 0.25),
                    "protein_g": round(protein_g * 0.25),
                    "carbs_g": round(carbs_g * 0.25),
                    "fats_g": round(fat_g * 0.30),
                    "estimated_cost_inr": round(daily_budget * 0.25),
                }
            ]

        # 4, 5, or 6 Spaced Meals
        return [
            {
                "type": "Breakfast",
                "time": "07:30 AM",
                "title": "Fast Bowler High-Protein Energy Bowl",
                "items": [
                    get_breakfast_protein(),
                    "100g Rolled Oats with Protein Isolate" if not has_gluten_allergy else "100g Rice Flakes with Protein",
                    "1 Banana with Chia Seeds",
                    "250ml Electrolyte Fresh Juice"
                ],
                "calories": round(total_cals * 0.24),
                "protein_g": round(protein_g * 0.26),
                "carbs_g": round(carbs_g * 0.25),
                "fats_g": round(fat_g * 0.24),
                "estimated_cost_inr": round(daily_budget * 0.25),
            },
            {
                "type": "Pre-Spell Fuel",
                "time": "10:30 AM",
                "title": "Explosive Ground Session Primer",
                "items": [
                    "2 Multigrain Toast slices with Raw Honey" if not has_gluten_allergy else "2 Boiled Potatoes with Rock Salt",
                    get_nuts_item(),
                    "Black Coffee or Pre-workout electrolyte drink"
                ],
                "calories": round(total_cals * 0.12),
                "protein_g": round(protein_g * 0.08),
                "carbs_g": round(carbs_g * 0.16),
                "fats_g": round(fat_g * 0.12),
                "estimated_cost_inr": round(daily_budget * 0.12),
            },
            {
                "type": "Post-Spell Recovery",
                "time": "01:00 PM",
                "title": "Kinetic Recovery & Glycogen Replenishment",
                "items": [
                    "1 Scoop Isolate Protein in Fresh Coconut Water",
                    "1 Large Sweet Potato or Ripe Banana"
                ],
                "calories": round(total_cals * 0.12),
                "protein_g": round(protein_g * 0.18),
                "carbs_g": round(carbs_g * 0.14),
                "fats_g": round(fat_g * 0.06),
                "estimated_cost_inr": round(daily_budget * 0.18),
            },
            {
                "type": "Lunch",
                "time": "02:30 PM",
                "title": "Athletic Performance & Repair Plate",
                "items": [
                    get_lunch_protein(),
                    "200g Brown Basmati Rice / Quinoa",
                    "Mixed Steamed Veggies (Broccoli, Carrots)",
                    "1 Bowl Fresh Greek Yogurt / Curd" if not has_dairy_allergy and not is_vegan else "Sprouted Lentil Salad"
                ],
                "calories": round(total_cals * 0.26),
                "protein_g": round(protein_g * 0.26),
                "carbs_g": round(carbs_g * 0.23),
                "fats_g": round(fat_g * 0.26),
                "estimated_cost_inr": round(daily_budget * 0.25),
            },
            {
                "type": "Evening Snack",
                "time": "06:00 PM",
                "title": "Workload Recovery Snack",
                "items": [
                    "Roasted Bengal Gram (Chana) - 60g",
                    "Handful of Crushed Walnuts & Pumpkin Seeds" if not has_nut_allergy else "Roasted Makhana & Pumpkin Seeds",
                    "Green Tea with Squeezed Lemon"
                ],
                "calories": round(total_cals * 0.08),
                "protein_g": round(protein_g * 0.06),
                "carbs_g": round(carbs_g * 0.08),
                "fats_g": round(fat_g * 0.10),
                "estimated_cost_inr": round(daily_budget * 0.08),
            },
            {
                "type": "Dinner",
                "time": "08:30 PM",
                "title": "Overnight Tissue Repair & Lumbar Care",
                "items": [
                    get_dinner_protein(),
                    get_carb_base(),
                    "Generous Green Spinach & Cucumber Salad",
                    "1 Glass Warm Golden Turmeric Milk" if not has_dairy_allergy and not is_vegan else "Warm Golden Oat Milk"
                ],
                "calories": round(total_cals * 0.18),
                "protein_g": round(protein_g * 0.16),
                "carbs_g": round(carbs_g * 0.14),
                "fats_g": round(fat_g * 0.22),
                "estimated_cost_inr": round(daily_budget * 0.12),
            },
        ]

    # -----------------------------------------------------------------
    # 2. WORKLOAD MANAGEMENT (WRC / ACWR)
    # -----------------------------------------------------------------

    def calculate_workload(
        self,
        sessions: List[Dict[str, Any]],
        reference_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Calculates training load using the session-RPE method (Foster et al.)
        and computes the Acute:Chronic Workload Ratio (ACWR).
        """
        if not sessions:
            return {
                "success": False,
                "reason": "No session data provided.",
            }

        parsed_sessions = self._parse_and_score_sessions(sessions)

        ref_date = (
            datetime.strptime(reference_date, "%Y-%m-%d").date()
            if reference_date
            else max(s["date"] for s in parsed_sessions)
        )

        acute_load = self._average_daily_load(
            parsed_sessions, ref_date, config.ACUTE_WINDOW_DAYS
        )
        chronic_load = self._average_daily_load(
            parsed_sessions, ref_date, config.CHRONIC_WINDOW_DAYS
        )

        if chronic_load == 0:
            acwr = None
        else:
            acwr = round(acute_load / chronic_load, 2)

        prev_week_ref = ref_date - timedelta(days=7)
        prev_week_load = self._average_daily_load(
            parsed_sessions, prev_week_ref, config.ACUTE_WINDOW_DAYS
        )

        trend = self._determine_trend(acute_load, prev_week_load)
        zone, recommendation = self._interpret_acwr(acwr)

        return {
            "success": True,
            "reference_date": ref_date.isoformat(),
            "acute_load_7d_avg": round(acute_load, 1),
            "chronic_load_28d_avg": round(chronic_load, 1),
            "acwr": acwr,
            "zone": zone,
            "trend": trend,
            "recommendation": recommendation,
            "disclaimer": (
                "This is a general training-load monitoring guideline, "
                "not a medical assessment or injury prediction. Consult "
                "a qualified physiotherapist/S&C coach for individual decisions."
            ),
        }

    def generate_guidance_text(
        self,
        workload_result: Dict[str, Any],
        llm_client: Optional[Any] = None,
    ) -> str:
        """
        Turns workload numbers into a concise natural-language guidance note.
        """
        if not workload_result.get("success"):
            return workload_result.get("reason", "No workload data.")

        client = llm_client or LLMClient()

        system_prompt = (
            "You are a cricket strength-and-conditioning assistant. You "
            "will be given ALREADY-CALCULATED training-load numbers "
            "(acute:chronic workload ratio and trend). Do not recalculate "
            "or invent numbers. Write 2-3 short, practical sentences of "
            "guidance for the player based only on the given numbers and "
            "zone. Do not make medical or injury-diagnosis claims -- "
            "frame this as training-load awareness only. Respond with "
            "plain text, no JSON, no markdown."
        )

        user_prompt = (
            f"ACWR: {workload_result['acwr']}, zone: {workload_result['zone']}, "
            f"trend: {workload_result['trend']}, "
            f"acute load (7d avg): {workload_result['acute_load_7d_avg']}, "
            f"chronic load (28d avg): {workload_result['chronic_load_28d_avg']}."
        )

        try:
            return client.generate_text(system_prompt, user_prompt)
        except Exception:
            return workload_result["recommendation"]

    # -----------------------------------------------------------------
    # AI SPORTS-SCIENCE COACHING ASSESSMENT (LLM + BMR + TDEE + ACWR)
    # -----------------------------------------------------------------

    def generate_ai_coaching_assessment(
        self,
        athlete_profile: Dict[str, Any],
        nutrition_metrics: Dict[str, Any],
        workload_metrics: Dict[str, Any],
        llm_client: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Synthesizes BMR, TDEE, ACWR Workload Ratio, and Macro targets into an elite
        AI sports-science coaching evaluation detailing:
        - Current Positives & Strengths
        - Current Lacks, Deficits & Vulnerabilities
        - Required Physiological & Nutritional Needs
        - Action Directives tailored specifically to the athlete's Goal
        - Coach's Personal Summary Note
        """
        weight = float(athlete_profile.get("weight", athlete_profile.get("weight_kg", 68.0)))
        height = float(athlete_profile.get("height", athlete_profile.get("height_cm", 178.0)))
        age = int(athlete_profile.get("age", 20))
        goal = str(athlete_profile.get("goal", "increase_speed"))
        food_pref = str(athlete_profile.get("foodPref", athlete_profile.get("food_preference", "vegetarian")))
        status = str(athlete_profile.get("athlete_status", "student"))

        height_m = max(1.2, height / 100.0)
        bmi = round(weight / (height_m ** 2), 1)

        bmr = round(float(nutrition_metrics.get("bmr", 1680)))
        tdee = round(float(nutrition_metrics.get("tdee", nutrition_metrics.get("base_tdee", 2900))))
        target_cals = round(float(nutrition_metrics.get("target_calories", nutrition_metrics.get("training_day_calories", 3200))))
        protein_g = round(float(nutrition_metrics.get("protein_g", nutrition_metrics.get("daily_protein_target_g", round(weight * 2.2)))))
        carbs_g = round(float(nutrition_metrics.get("carbs_g", nutrition_metrics.get("daily_carbs_target_g", round((target_cals * 0.50) / 4)))))
        fats_g = round(float(nutrition_metrics.get("fats_g", nutrition_metrics.get("daily_fats_target_g", round((target_cals * 0.25) / 9)))))

        acwr = round(float(workload_metrics.get("acwr_score", workload_metrics.get("acwr", 1.18))), 2)
        acwr_status = str(workload_metrics.get("acwr_status", "Optimal Range (1.10 - 1.30 ACWR)"))
        weekly_overs = int(workload_metrics.get("total_weekly_overs", workload_metrics.get("weekly_overs", 16)))
        gym_days = int(workload_metrics.get("gym_days", 3))
        ground_days = int(workload_metrics.get("ground_days", 4))
        max_overs_days = int(workload_metrics.get("max_overs_days", 2))

        client = llm_client or LLMClient()

        system_prompt = (
            "You are PacePro AI's Chief Sports Scientist and Fast Bowling Performance Director. "
            "You are provided with pre-calculated, verified sports-science metrics (BMI, BMR, TDEE, ACWR Workload Ratio, Macro targets, and fast bowling session loads). "
            "Do NOT invent or recalculate new numbers. Your job is to analyze these numbers and deliver an authoritative, high-impact athletic coaching diagnosis.\n\n"
            "CRITICAL RULES FOR 'positives':\n"
            "Positives MUST ONLY highlight physical baseline, body composition, training split, and workload conditioning strengths:\n"
            f"1. Athletic BMI & Power-to-weight baseline (e.g. 'Healthy athletic BMI of {bmi} kg/m² provides a solid power-to-weight baseline for explosive run-up acceleration without excessive joint loading.').\n"
            f"2. Balanced ~4–5 hours weekly training split between gym strength conditioning ({gym_days} days) and ground bowling ({ground_days} days).\n"
            f"3. Calibrated ACWR workload conditioning ({acwr}) in the safe adaptation zone ({acwr_status}) building bowling stamina.\n\n"
            "STRICT NEGATIVE CONSTRAINT: DO NOT mention BMR calories (e.g. 'High basal metabolic rate'), DO NOT state calorie surplus numbers (e.g. 'Prescribed intake exceeds maintenance'), and DO NOT list raw protein targets (e.g. 'Protein target of 136g') in the positives list. All dietary and calorie numbers belong exclusively in 'required_needs'.\n\n"
            "Respond ONLY with a valid JSON object matching this exact structure, with no markdown code fences:\n"
            "{\n"
            '  "positives": ["Healthy athletic BMI & power-to-weight baseline", "Well-structured ~4-5 hours gym and ground split", "Calibrated ACWR workload conditioning ratio"],\n'
            '  "current_lacks": ["Metabolic or workload deficit 1", "Deficit 2", "Deficit 3"],\n'
            '  "required_needs": {\n'
            '    "daily_calories": "Precise calorie prescription for bowling vs recovery days",\n'
            '    "protein_target": "Exact protein grams and per-meal leucine threshold",\n'
            '    "carbs_and_fueling": "Pre-spell glycogen loading and intra-session hydration",\n'
            '    "recovery_and_joint_care": "Spinal disc protection, sleep duration, and anti-inflammatory micronutrients"\n'
            '  },\n'
            '  "goal_action_directives": ["1. Action one", "2. Action two", "3. Action three", "4. Action four"],\n'
            '  "coach_summary": "Direct, motivational coaching summary highlighting the primary focal point."\n'
            "}"
        )

        user_prompt = (
            f"ATHLETE PROFILE:\n"
            f"- Age: {age} yrs | Height: {height} cm | Bodyweight: {weight} kg | BMI: {bmi} kg/m² | Status: {status}\n"
            f"- Stated Athletic Goal: {goal.replace('_', ' ').title()}\n"
            f"- Diet Preference: {food_pref.title()}\n\n"
            f"CALCULATED SPORTS-SCIENCE METRICS:\n"
            f"- BMI: {bmi} kg/m² (Healthy fast-bowling athletic baseline)\n"
            f"- BMR (Mifflin-St Jeor): {bmr} kcal\n"
            f"- TDEE (Maintenance Energy Expenditure): {tdee} kcal\n"
            f"- Prescribed Athletic Intake: {target_cals} kcal/day\n"
            f"- Macros: Protein {protein_g}g (2.2g/kg), Carbs {carbs_g}g, Fats {fats_g}g\n"
            f"- Workload Ratio (ACWR): {acwr} ({acwr_status})\n"
            f"- Weekly Bowling Volume: {weekly_overs} overs ({ground_days} ground sessions, {max_overs_days} max-overs match spells, {gym_days} gym days, ~4-5 hours total training)\n\n"
            f"Please generate the coaching assessment for '{goal}'. Ensure 'positives' contains ONLY athletic BMI, the ~4-5 hours gym/ground split, and ACWR workload tolerance."
        )

        try:
            parsed = client.generate_json(system_prompt, user_prompt)
            if (
                isinstance(parsed, dict) 
                and "positives" in parsed 
                and "current_lacks" in parsed 
                and "required_needs" in parsed 
                and "goal_action_directives" in parsed
            ):
                # Cleanse any accidental calorie/BMR leaks in positives
                clean_positives = []
                for p in parsed.get("positives", []):
                    p_str = str(p)
                    lower_p = p_str.lower()
                    if any(term in lower_p for term in ["bmr", "metabolic rate", "prescribed intake", "exceeds maintenance", "kcal", "protein target of"]):
                        continue
                    clean_positives.append(p_str)
                
                # If filtered list is less than 3, replenish with clean athletic baseline
                if len(clean_positives) < 3:
                    if not any("bmi" in cp.lower() for cp in clean_positives):
                        clean_positives.insert(0, f"Healthy athletic BMI ({bmi} kg/m²) — solid lean foundation and healthy power-to-weight ratio for explosive run-up acceleration without excess joint load.")
                    if not any("split" in cp.lower() or "hour" in cp.lower() for cp in clean_positives):
                        clean_positives.append(f"Well-structured training split: balanced ~4–5 hours weekly distribution between gym strength conditioning ({gym_days} days) and explosive ground bowling ({ground_days} days).")
                    if not any("acwr" in cp.lower() or "workload" in cp.lower() for cp in clean_positives):
                        clean_positives.append(f"Calibrated ACWR workload ({acwr}) in the '{acwr_status}', maintaining peak match-over conditioning while ensuring safe recovery intervals.")

                parsed["positives"] = clean_positives[:3]
                parsed["source"] = "llm_analysis"
                parsed["status"] = "verified"
                return parsed
        except Exception as e:
            logger.warning(f"[PlannerService] LLM coaching assessment unavailable, generating deterministic report: {e}")

        # Deterministic High-Performance Fallback Report
        return self._build_fallback_coaching_assessment(
            weight=weight,
            height=height,
            bmi=bmi,
            goal=goal,
            food_pref=food_pref,
            bmr=bmr,
            tdee=tdee,
            target_cals=target_cals,
            protein_g=protein_g,
            carbs_g=carbs_g,
            fats_g=fats_g,
            acwr=acwr,
            acwr_status=acwr_status,
            weekly_overs=weekly_overs,
            ground_days=ground_days,
            gym_days=gym_days,
            max_overs_days=max_overs_days
        )

    def _build_fallback_coaching_assessment(
        self,
        weight: float,
        height: float,
        bmi: float,
        goal: str,
        food_pref: str,
        bmr: int,
        tdee: int,
        target_cals: int,
        protein_g: int,
        carbs_g: int,
        fats_g: int,
        acwr: float,
        acwr_status: str,
        weekly_overs: int,
        ground_days: int,
        gym_days: int,
        max_overs_days: int
    ) -> Dict[str, Any]:
        is_veg = "veg" in food_pref.lower() and "non" not in food_pref.lower()
        
        # Athletic Baseline & Positive Insights
        if 18.5 <= bmi <= 24.9:
            bmi_text = f"Optimal athletic BMI ({bmi} kg/m²) — solid lean foundation and healthy power-to-weight ratio for explosive run-up acceleration without excess joint load."
        elif bmi < 18.5:
            bmi_text = f"Agile athletic frame (BMI {bmi} kg/m²) — high baseline mobility; planned caloric surplus will convert directly into fast-bowling lean power."
        else:
            bmi_text = f"Robust athletic mass (BMI {bmi} kg/m²) — excellent momentum foundation; strength training will harness this mass for high ball release velocity."

        positives = [
            bmi_text,
            f"Well-structured training split: balanced ~4–5 hours weekly distribution between gym strength conditioning ({gym_days} days) and explosive ground bowling ({ground_days} days).",
            f"Calibrated ACWR workload ({acwr}) in the '{acwr_status}', maintaining peak match-over conditioning while ensuring safe recovery intervals between bowling spells."
        ]

        current_lacks = [
            f"Intra-session glycogen deficit: Dual ground/gym days burn ~{round(tdee * 0.4)} kcal in 3 hours; failing to intake fast carbs 45 min pre-spell leads to late-spell velocity drop.",
            f"Protein distribution gap: As a {food_pref} athlete, you need 3.0g of leucine per feeding; ensure your {protein_g}g is evenly distributed across 4-5 spaced meals rather than 2 large ones." if is_veg else f"Hydration & mineral deficit: High bowling volume ({weekly_overs} overs) expels 1.5–2L sweat; sodium/potassium replacement is critical to prevent quad/calf cramping.",
            f"Spinal shear risk on high-intensity days: {max_overs_days} max-overs spells generate ground reaction forces up to 8–9x bodyweight on the front leg brace."
        ]

        required_needs = {
            "daily_calories": f"{target_cals} kcal on bowling days (+350 kcal surplus for tendon regeneration); scale back to ~{max(2200, target_cals - 450)} kcal on full rest days.",
            "protein_target": f"{protein_g}g daily (2.2g per kg of bodyweight). Consume 30-35g within 30 minutes post-session with fast-acting carbohydrates.",
            "carbs_and_fueling": f"{carbs_g}g complex carbs (oats, brown basmati rice, sweet potatoes) to keep muscle glycogen reserves saturated before match spells.",
            "recovery_and_joint_care": f"{(weight * 0.05):.1f} - {(weight * 0.06):.1f} Liters of electrolyte water daily, plus 1000mg Calcium & 2000 IU Vitamin D3 for lumbar disc and tibia bone density."
        }

        goal_clean = goal.replace("_", " ").lower()
        if "speed" in goal_clean or "pace" in goal_clean or "performance" in goal_clean:
            directives = [
                "1. Pre-Spell Priming: Ingest 40-50g fast carbohydrates (banana + honey toast or isotonic drink) 45 min before bowling to maximize kinetic power output in overs 5–8.",
                "2. Front Knee Brace Stiffness: Pair your 18° knee plant with post-bowling isometric wall sits (3x60s) and a high-leucine recovery shake within 30 min.",
                "3. 48-Hour Spell Spacing: Strictly maintain at least 48 hours between your max-overs spells to keep ACWR below 1.35 and prevent lumbar stress reactions.",
                "4. Sleep Architecture: Target 8.5 hours of uninterrupted sleep nightly—peak testosterone and growth hormone spikes occur during deep slow-wave sleep."
            ]
            summary = f"Your metabolic engine is calibrated for explosive pace. Prioritize pre-spell glycogen saturation and strict 48h spell intervals to hit top velocity while protecting your lower back."
        elif "muscle" in goal_clean or "hypertrophy" in goal_clean:
            directives = [
                f"1. Progressive Caloric Surplus: Maintain {target_cals} kcal consistently across all 7 days to fuel hypertrophy without compromising bowling mobility.",
                "2. Hyper-aminoacidemia Cadence: Consume 35g protein every 3.5 hours to maximize 24-hour muscle protein synthesis (MPS).",
                "3. Compound Loading: Schedule heavy compound lifts (Trap Bar Deadlifts, Front Squats) on non-bowling days with a 7-8 hour gap from ground work.",
                "4. Creatine Saturation: Supplement 3-5g Creatine Monohydrate daily to increase intramuscular phosphocreatine and explosive run-up force."
            ]
            summary = f"Your surplus of {target_cals} kcal and {protein_g}g protein will build robust athletic armor to withstand repetitive bowling delivery impacts."
        else:
            directives = [
                f"1. Energy Equilibrium: Target {target_cals} kcal to maintain lean power-to-weight ratio for swift run-up acceleration.",
                f"2. High Protein Satiety: Hit {protein_g}g protein daily to safeguard lean muscle mass while metabolizing excess adipose tissue.",
                "3. Hydration Electrolytes: Drink at least 3.5L of water supplemented with electrolytes during bowling spells.",
                "4. Active Recovery Protocols: Incorporate foam rolling, hamstring mobility, and contrast showers on rest days."
            ]
            summary = f"Balanced protocol tailored to sustain high kinetic output, optimize body composition, and maintain spinal resilience."

        return {
            "source": "sports_science_engine",
            "status": "calculated",
            "positives": positives,
            "current_lacks": current_lacks,
            "required_needs": required_needs,
            "goal_action_directives": directives,
            "coach_summary": summary
        }

    # -----------------------------------------------------------------
    # 3. COMBINED WORKLOAD & NUTRITION PROTOCOL
    # -----------------------------------------------------------------

    def generate_combined_plan(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates a synchronized 7-day cricket fast-bowling workload & nutrition plan
        adapting to Student vs Full-Time Athlete status, gym volume, ground volume,
        and max-overs spells with ACWR injury risk monitoring and ECB directives.
        """
        athlete_status = payload.get("athlete_status", "student")
        gym_days = int(payload.get("gym_days", payload.get("gym_days_per_week", 3)))
        ground_days = int(payload.get("ground_days", payload.get("ground_days_per_week", 4)))
        weekly_bowling_days = int(payload.get("weekly_bowling_days", payload.get("max_overs_days", 3)))
        max_overs_per_spell = int(payload.get("max_overs_per_spell", payload.get("per_max_over", 8)))
        avg_overs_per_session = int(payload.get("avg_overs_per_session", payload.get("average_overs", 5)))
        max_speed = float(payload.get("max_speed", 132.0))
        injury_status = str(payload.get("injury_status", payload.get("current_injury_status", "none"))).strip()
        session_gap = payload.get("session_gap", "Morning Ground (6:30 AM) • Evening Gym (5:30 PM)")
        user_weight = float(payload.get("weight", payload.get("weight_kg", 68.0)))
        user_height = float(payload.get("height", payload.get("height_cm", 175.0)))
        user_age = int(payload.get("age", 20))
        user_sex = payload.get("sex", "male").lower()
        goal = payload.get("goal", "increase_speed")
        weekly_budget = float(payload.get("budgetWeekly", payload.get("budget", {}).get("amount", 3500)))
        raw_allergies = payload.get("allergies", [])
        selected_allergies = [str(a).lower() for a in raw_allergies] if isinstance(raw_allergies, list) else []

        max_overs_days = int(payload.get("max_overs_days", min(weekly_bowling_days, 2)))

        # BMR & TDEE Calculations (Mifflin-St Jeor)
        bmr = self.calculate_bmr(user_weight, user_height, user_age, user_sex)
        total_sessions = gym_days + ground_days
        act_multiplier = (
            1.85 if total_sessions >= 8
            else 1.725 if total_sessions >= 6
            else 1.55 if total_sessions >= 4
            else 1.4
        )
        base_tdee = round(bmr * act_multiplier)

        target_cals = base_tdee
        if "muscle" in goal or "hypertrophy" in goal:
            target_cals += 350
        elif "fat_loss" in goal or "cut" in goal:
            target_cals -= 450
        else:  # increase_speed / performance
            target_cals += 400

        base_protein = round(user_weight * 2.2)

        # Workload & Overs Analytics
        # Estimated total overs = (bowling days * avg overs) + any max spell overshoot
        estimated_weekly_overs = round((weekly_bowling_days * avg_overs_per_session) + max(0, max_overs_per_spell - avg_overs_per_session))
        overs_ceiling = 18 if user_age < 19 else (24 if user_age < 23 else 30)
        spell_ceiling = 6 if user_age < 19 else 8
        is_overs_overloaded = estimated_weekly_overs > overs_ceiling
        is_spell_overloaded = max_overs_per_spell > spell_ceiling
        is_back_to_back = "back-to-back" in session_gap.lower() or "< 1 hour" in session_gap.lower()
        has_active_injury = injury_status.lower() not in ("none", "fully fit", "fully fit (no niggles)", "fit", "")
        is_injury_speed_clash = has_active_injury and max_speed >= 125

        # ACWR calculation factoring in volume, safe ceiling ratio, and compression strain
        acwr_score = round(0.92 + (estimated_weekly_overs / (overs_ceiling * 1.35)) * 0.45 + (0.15 if is_back_to_back else 0.0), 2)
        acwr_status = (
            "High Spike Alert (>1.40 ACWR: Monitor Spinal Soreness)"
            if acwr_score > 1.40
            else ("Elevated Strain (1.30 - 1.40 ACWR)" if acwr_score > 1.30 else "Optimal Sweet Spot (0.80 - 1.30 ACWR)")
        )

        ecb_check = (
            f"Warning: {estimated_weekly_overs} overs/wk exceeds safe ceiling ({overs_ceiling} overs for age {user_age})."
            if is_overs_overloaded
            else f"Compliant: Within safe weekly bowling ceiling ({overs_ceiling} overs max)."
        )

        # Important Fast-Bowler Biomechanics & Lifestyle Inputs
        bowling_arm = str(payload.get("bowling_arm", "right")).lower()
        bowling_action = str(payload.get("bowling_action", "semi_open")).lower()
        runup_length = str(payload.get("runup_length", "medium")).lower()
        sleep_hours = float(payload.get("sleep_hours", payload.get("sleepHours", 7.5)))
        water_intake_liters = float(payload.get("water_intake_liters", payload.get("waterIntake", 3.0)))
        supplement_preference = str(payload.get("supplement_preference", payload.get("supplementPref", "basic"))).lower()
        meal_frequency = int(payload.get("meal_frequency", payload.get("mealFrequency", 5)))

        # Strength Baseline Capacity & Training Environment
        strength_track = str(payload.get("strength_track", "gym_lifting" if gym_days > 0 else "calisthenics_ground")).lower()
        has_gym = gym_days > 0 and "calisthenics" not in strength_track and "bodyweight" not in strength_track
        does_deadlifts = bool(payload.get("does_deadlifts", True)) if has_gym else False

        deadlift_kg = float(payload.get("deadlift_kg", payload.get("trap_bar_deadlift_kg", 85.0)))
        deadlift_reps = int(payload.get("deadlift_reps", 5))
        split_squat_kg = float(payload.get("split_squat_kg", 14.0))
        split_squat_reps = int(payload.get("split_squat_reps", 8))
        pullups_reps = int(payload.get("pullups_reps", 6))
        plank_seconds = int(payload.get("plank_seconds", 60))

        # Calisthenics & Bodyweight Metrics (For bowlers without gym or deadlifts)
        bodyweight_squats_reps = int(payload.get("bodyweight_squats_reps", 30))
        pushups_reps = int(payload.get("pushups_reps", 20))
        single_leg_bw_squats = int(payload.get("single_leg_bw_squats", split_squat_reps if not has_gym else 12))

        height_m = max(1.2, user_height / 100.0)
        bmi = round(user_weight / (height_m * height_m), 1)
        est_deadlift_1rm = round(deadlift_kg * (1.0 + (deadlift_reps / 30.0)), 1) if does_deadlifts else 0.0
        strength_ratio = round(est_deadlift_1rm / max(35.0, user_weight), 2) if does_deadlifts else 0.0

        is_mixed_action = "mixed" in bowling_action
        is_sleep_deprived = sleep_hours < 6.5
        is_dehydrated = water_intake_liters < 3.5
        is_severely_dehydrated = water_intake_liters < 2.5
        is_underweight = bmi < 18.5
        is_overweight = bmi > 26.0
        is_strength_deficient = does_deadlifts and strength_ratio < 1.40
        is_calisthenics_deficient = (not has_gym or not does_deadlifts) and (pushups_reps < 15 or bodyweight_squats_reps < 20)

        # -------------------------------------------------------------
        # 1. OVERLOAD, PHYSIOLOGICAL & STRENGTH WARNINGS
        # -------------------------------------------------------------
        overload_warnings = []

        # A. Hydration / Pani Warning (Highest Priority for Spine & Cramps)
        if is_dehydrated:
            overload_warnings.append({
                "id": "dehydration_disc_thinning",
                "type": "Hydration & Disc Thinning Hazard",
                "severity": "danger" if is_severely_dehydrated else "warning",
                "title": f"🚨 Critical Dehydration Vulnerability ({water_intake_liters}L/Day vs 3.5L+ Target)",
                "reason": f"Aap daily sirf {water_intake_liters}L paani pee rahe hain. Fast bowling me 1.5–2.0L sweat per spell nikalta hai. Itna kam paani intervertebral lumbar disc hydration maintain karne ke liye dangerous level par kam hai.",
                "physiological_risk": "Dehydrated lumbar discs lose 15-20% shock-absorbing height under 8-9x bodyweight delivery strides, accelerating nerve root compression, severe quad/calf cramps, and pars stress fractures.",
                "actionable_fix": f"Daily fluid intake ko kam se kam {max(3.8, round(user_weight * 0.055, 1))}L tak turant badhayein, aur spell me har 2 overs baad 150-200ml electrolyte hydration (sodium + potassium) zaroor lein."
            })

        # B. BMI & Muscular Armor Warning
        if is_underweight:
            overload_warnings.append({
                "id": "underweight_muscular_armor_deficit",
                "type": "Low Muscular Armor & Joint Impact Vulnerability",
                "severity": "danger",
                "title": f"⚠️ Underweight & Muscular Armor Deficit (BMI: {bmi})",
                "reason": f"Aapka BMI {bmi} hai (Underweight category for fast bowling). Front-foot contact par lead knee aur lower back par 8-9x bodyweight ground reaction force aati hai. Muscle mass aur posterior armor na hone se bones aur lumbar joints direct shock absorb karte hain.",
                "physiological_risk": "High risk of tibial stress fractures, early knee joint degeneration, and inability to maintain a stiff front-knee brace without spinal shear.",
                "actionable_fix": (
                    "Staged Target Phase 1: First gain 3-5 kg lean muscular armor with a slight calorie surplus and progressive compound loading before pushing for extreme bowling pace."
                    if has_gym else
                    "Staged Target Phase 1: First gain 3-5 kg lean muscular armor with calorie surplus, protein intake, and progressive bodyweight calisthenics (Nordic hamstring drops, pushups, pistol step-ups)."
                )
            })
        elif is_overweight:
            overload_warnings.append({
                "id": "elevated_joint_impact_load",
                "type": "Elevated Impact Joint Load",
                "severity": "warning",
                "title": f"⚠️ Elevated Joint Impact Load (BMI: {bmi})",
                "reason": f"Aapka BMI {bmi} hai. Higher body mass front foot contact par ground reaction brake force ko exponentially badha deta hai.",
                "physiological_risk": "Excessive mechanical stress on patellar tendon and ankle ligaments during delivery stride deceleration.",
                "actionable_fix": "Athletic body recomposition: Maintain high strength training while trimming body fat percentage through clean conditioning and explosive drills."
            })

        # C. Strength Baseline & Knee Brace Capacity Warning
        if does_deadlifts and is_strength_deficient:
            overload_warnings.append({
                "id": "posterior_chain_strength_deficit",
                "type": "Posterior Chain & Brake Force Deficit",
                "severity": "warning",
                "title": f"⚠️ Posterior Chain Strength Deficit ({strength_ratio}x Bodyweight Deadlift)",
                "reason": f"Aapka estimated Deadlift 1RM ({est_deadlift_1rm} kg) bodyweight ka sirf {strength_ratio}x hai. Fast bowling me front knee brace ko firm lock rakhne ke liye kam se kam 1.5x - 1.8x BW strength mandatory hoti hai.",
                "physiological_risk": "Lead leg buckles into excessive knee flexion (>30°) at release point, leaking 6-10 km/h ball speed and transferring axial shear torque directly onto the lower back.",
                "actionable_fix": "Progressive overload on Trap Bar Deadlifts (3 sets × 5 reps @ 80-85% 1RM) and Bulgarian Split Squats twice a week to build eccentric brake force."
            })
        elif not has_gym or not does_deadlifts:
            if is_calisthenics_deficient:
                overload_warnings.append({
                    "id": "calisthenics_armor_deficit",
                    "type": "Bodyweight Muscular Armor Deficit",
                    "severity": "warning",
                    "title": "⚠️ Bodyweight Armor Deficit (Pushups < 15 or Squats < 20)",
                    "reason": "Gym/Deadlifts na karne par fast bowling ke 8-9x BW delivery stride shock ko absorb karne ke liye high bodyweight calisthenic armor (pushups, single-leg squats, glute bridges) zaroori hai.",
                    "physiological_risk": "Lack of eccentric braking strength causes knee collapse and spinal jarring at ball release.",
                    "actionable_fix": "Daily bodyweight routine start karein: 3 sets × Isometric Wall Sits (45-60s), Single-Leg Chair Step-ups (3x12), and Nordic Hamstring Drops with partner/bed anchor."
                })

        # D. Mixed Action Spinal Hazard
        if is_mixed_action:
            overload_warnings.append({
                "id": "mixed_action_counter_rotation",
                "type": "Biomechanical Spinal Hazard",
                "severity": "danger",
                "title": "Severe Counter-Rotation Spinal Shear (Mixed Action Detected)",
                "reason": "Aapka bowling action 'Mixed Action' hai (hips aur shoulders opposite directions me counter-rotate karte hain). Fast bowling me ye L4-L5 lumbar stress fracture (spondylolysis) ka #1 cause hai.",
                "physiological_risk": "Counter-rotation generates 2.5x shear torsion directly onto the contralateral lumbar pars interarticularis during back-foot impact.",
                "actionable_fix": "Coaching drill: Back-foot landing angle ko bowling crease ke parallel align karein aur shoulders ko hips ke saath synchronize karein (Adopt pure side-on ya front-on)."
            })

        # E. Sleep Architecture Deficit
        if is_sleep_deprived:
            overload_warnings.append({
                "id": "sleep_collagen_deficit",
                "type": "Cellular Recovery Impairment",
                "severity": "danger" if sleep_hours < 5.5 else "warning",
                "title": f"Sleep Architecture Deficit ({sleep_hours} Hours/Night)",
                "reason": f"Hafte me {weekly_bowling_days} bowling days ke saath sirf {sleep_hours} ghante sona muscle aur tendon collagen remodeling ko rok deta hai.",
                "physiological_risk": "Deep slow-wave sleep me growth hormone (GH) secretion aur osteoblast bone remodeling suppress ho jati hai, increasing stress injury probability by 1.7x.",
                "actionable_fix": "Nightly sleep duration ko 8-8.5 hours tak elevate karein aur post-bowling 20-min afternoon nap schedule karein."
            })

        # F. Acute Overs Volume Overload
        if is_overs_overloaded:
            overload_warnings.append({
                "id": "overs_volume_spike",
                "type": "Acute Bowling Volume Overload",
                "severity": "danger" if estimated_weekly_overs > overs_ceiling + 6 else "warning",
                "title": f"Excessive Weekly Overs Spike ({estimated_weekly_overs} Overs vs Safe Limit {overs_ceiling} Overs)",
                "reason": f"Aapka current weekly bowling volume ({estimated_weekly_overs} overs) age-{user_age} ECB safe ceiling ({overs_ceiling} overs) se zyada hai.",
                "physiological_risk": "Acute spike in ground reaction forces (8-9x bodyweight) leads to lumbar bone marrow edema, L4-L5 pars stress reactions, and tibial fatigue.",
                "actionable_fix": f"Weekly bowling sessions ko {weekly_bowling_days} days se kam karke {max(2, weekly_bowling_days - 1)} days karein, aur weekly total ko {overs_ceiling} overs par cap karein."
            })

        # G. Spell Intensity Spike
        if is_spell_overloaded:
            overload_warnings.append({
                "id": "spell_intensity_spike",
                "type": "Single Spell Overstimulation",
                "severity": "danger" if max_overs_per_spell >= 10 else "warning",
                "title": f"Dangerous Spell Ceiling ({max_overs_per_spell} Overs in Single Spell)",
                "reason": f"Ek continuous spell me {max_overs_per_spell} overs daalna fast bowler directives ({spell_ceiling} overs max) ko violate karta hai.",
                "physiological_risk": "Eccentric hamstring exhaustion causes trunk flexion collapse at release point, transferring 80% shear torque directly onto lumbar vertebrae.",
                "actionable_fix": f"Spell ko break karein: Maximum {spell_ceiling} overs ka spell rakhein, aur 2 spells ke beech minimum 3 ghante ka active rest lein."
            })

        # H. Back-to-Back Sessions
        if is_back_to_back:
            overload_warnings.append({
                "id": "session_compression",
                "type": "Spinal Compression Hazard",
                "severity": "danger",
                "title": "Severe Spinal Compression: Back-to-Back Gym & Ground Sessions",
                "reason": "Ground bowling delivery stride ke impact ke turant baad gym axial lifting (squats/deadlifts) karna spine ke liye critical risk hai.",
                "physiological_risk": "Intervertebral discs compress by 15-20% under bowling impacts. Immediate heavy lifting without disc rehydration causes disc bulges and facet syndrome.",
                "actionable_fix": "Ground bowling aur gym session ke beech kam se kam 6-8 ghante ka gap rakhein (e.g. 6:30 AM Ground, 5:30 PM Gym)."
            })

        # I. Active Injury Clash
        if is_injury_speed_clash:
            overload_warnings.append({
                "id": "injury_velocity_clash",
                "type": "High-Speed Trauma Collision",
                "severity": "danger",
                "title": f"High Velocity with Active Injury ({injury_status.replace('_', ' ').title()})",
                "reason": f"Active niggle/injury ({injury_status.replace('_', ' ')}) ke bawajood {max_speed} km/h par bowl karna injury ko aggravate kar raha hai.",
                "physiological_risk": "Micro-fracture progression (Spondylolysis), patellar tendon micro-tears, ya rotator cuff tendon impingement.",
                "actionable_fix": "Bowling intensity ko 65% spot bowling par limit karein. High-velocity match spells ko deload period khatam hone tak suspend karein."
            })

        # J. Insufficient Days Buffer
        if weekly_bowling_days >= 5:
            overload_warnings.append({
                "id": "tendon_remodeling_deficit",
                "type": "Inadequate Collagen Remodeling",
                "severity": "warning",
                "title": f"Insufficient Tendon Recovery Buffer ({weekly_bowling_days} Bowling Days / Week)",
                "reason": "Fast bowler ke patellar tendon aur rotator cuff ko collagen remodeling aur matrix repair ke liye 48 ghante rest chahiye hoti hai.",
                "physiological_risk": "Chronic patellar tendinopathy ('jumper's knee') aur shin splint stress reactions.",
                "actionable_fix": "Bowling days ko hafte me 3-4 days par limit karein, aur alternate days par active mobility aur pool recovery karein."
            })

        # -------------------------------------------------------------
        # 2. CURRENT MISTAKES & GAPS ("Aap Current Mein Kya Ghalat Kar Rahe Hain")
        # -------------------------------------------------------------
        current_mistakes_and_gaps = [
            {
                "category": "Workload & Bowling Volume",
                "mistake": (
                    f"Acute Overs Spikes ({estimated_weekly_overs} overs/wk vs {overs_ceiling} safe ceiling) with {max_overs_per_spell} overs max spell."
                    if is_overs_overloaded
                    else f"Irregular spell spacing: Bowling {weekly_bowling_days} days/week with variable spell densities."
                ),
                "why_harmful": "Bone matrix adaptation pace se bowling volume spike hone par pars interarticularis par stress fracture ka risk 3x badh jata hai.",
                "correction": f"Overs ko weekly {overs_ceiling} overs par lock karein aur spells ko structured 4-6 over burst sets me divide karein."
            },
            {
                "category": "Biomechanics & Action Alignment",
                "mistake": (
                    f"Bowling with a Mixed Action ({bowling_arm.title()}-Arm). Shoulders rotate counter to hips at crease."
                    if is_mixed_action
                    else f"{bowling_action.replace('_', ' ').title()} action with {runup_length} run-up approach."
                ),
                "why_harmful": "Shoulder-hip counter-rotation causes severe lateral flexion and hyperextension, concentrating 80% impact torque on contralateral facet joints." if is_mixed_action else "Suboptimal run-up deceleration creates excessive braking impact on the front knee.",
                "correction": "Adopt back-foot crease alignment parallel to stumps to prevent twisting." if is_mixed_action else "Maintain consistent approach rhythm and upright posture through release."
            },
            {
                "category": "Sleep & Cellular Regeneration",
                "mistake": (
                    f"Severe sleep deficit: Sleeping only {sleep_hours} hours/night with high bowling frequency."
                    if is_sleep_deprived
                    else f"Targeting {sleep_hours} hours of sleep without fixed circadian consistency."
                ),
                "why_harmful": "Slow-wave sleep suppression cuts growth hormone release in half, crippling tendon collagen cross-linking and bone density remodeling.",
                "correction": "Prioritize 8+ hours of uninterrupted sleep; eliminate blue light 60 mins before bed and supplement magnesium glycinate."
            },
            {
                "category": "Hydration & Disc Decompression",
                "mistake": (
                    f"Consuming only {water_intake_liters}L water daily despite high sweat rates."
                    if is_dehydrated
                    else f"Hydrating with plain water without electrolyte replenishment during spells."
                ),
                "why_harmful": "Dehydrated intervertebral discs lose viscoelastic shock absorption, exposing spinal nerve roots to severe impact compression.",
                "correction": f"Elevate baseline fluid intake to {(user_weight * 0.055):.1f}L/day with sodium, potassium, and magnesium salts during match spells."
            },
            {
                "category": "Session Spacing & Recovery",
                "mistake": (
                    f"Compressed session spacing: '{session_gap}'. Inadequate disc rehydration window."
                    if is_back_to_back
                    else "Ground aur gym sessions ke beech recovery nutrition aur posture resets skip karna."
                ),
                "why_harmful": "Spinal disc height (recovery) nahi hoti, jisse lower back stiffness aur front knee brace stability lose hoti hai.",
                "correction": "Ground session aur gym training ke beech 7-8 ghante ka gap rakhein ya gym alternate non-bowling days par shift karein."
            },
            {
                "category": "Fueling & Pre-Spell Glycogen",
                "mistake": f"Bowling at {max_speed} km/h without targeted 45-min pre-spell fast carbs.",
                "why_harmful": "Pace bowling phosphagen aur fast-glycolytic energy par chalti hai. 5th over ke baad ATP depletion se ball release speed 4-7 km/h drop hoti hai.",
                "correction": "Spell se 45 min pehle 40-50g fast carbs (banana + honey toast ya tender coconut water with electrolytes) consume karein."
            },
            {
                "category": "Protein & Leucine Synthesis",
                "mistake": f"Protein distribution gap: {supplement_preference.replace('_', ' ').title()} intake not hitting 3.0g leucine threshold every 3.5 hours.",
                "why_harmful": "Muscle protein synthesis (MPS) trigger karne ke liye har feeding me 3.0g leucine threshold zaroori hota hai.",
                "correction": f"Apne {base_protein}g protein ko {meal_frequency} balanced meals me distribute karein for continuous tendon repair."
            }
        ]

        # -------------------------------------------------------------
        # 3. FOOD MACROS: KITNA, KYUN & KYA KHANA HAI (3 PILLARS)
        # -------------------------------------------------------------
        food_pref = (payload.get("foodPref") or payload.get("food_preference") or "vegetarian").lower()
        is_veg = "veg" in food_pref and "non" not in food_pref

        has_allergy_peanuts = any("peanut" in a or "nut" in a for a in selected_allergies)
        has_allergy_dairy = any("dairy" in a or "lactose" in a or "milk" in a for a in selected_allergies)
        has_allergy_gluten = any("gluten" in a or "wheat" in a for a in selected_allergies)
        has_allergy_soy = any("soy" in a for a in selected_allergies)
        has_allergy_eggs = any("egg" in a for a in selected_allergies)

        # Determine curated protein sources honoring allergies and veg/non-veg
        if is_veg:
            veg_sources = []
            if not has_allergy_dairy:
                veg_sources.append("Paneer (150-200g)")
                veg_sources.append("Greek curd / probiotic dahi")
                veg_sources.append("Whey protein isolate")
            if not has_allergy_soy:
                veg_sources.append("Soya chunks (50g)")
                veg_sources.append("Tofu")
            veg_sources.extend(["Yellow moong dal", "Sprouted chickpeas (chana)", "Plant protein isolate"])
            protein_kya = ", ".join(veg_sources[:5])
        else:
            nonveg_sources = []
            if not has_allergy_eggs:
                nonveg_sources.append("4-5 Whole & white boiled eggs")
            nonveg_sources.append("Skinless grilled chicken breast (200g)")
            nonveg_sources.append("Grilled Rohu/Salmon fish")
            if not has_allergy_dairy:
                nonveg_sources.append("Low-fat paneer or Greek curd")
            nonveg_sources.append("Whey isolate")
            protein_kya = ", ".join(nonveg_sources[:5])

        # Healthy fats curated sources honoring nut allergies
        fat_sources = ["Cold-pressed mustard oil", "1 tbsp pure desi ghee", "Pumpkin & sunflower seeds", "Flaxseed powder"]
        if not has_allergy_peanuts:
            fat_sources.insert(0, "Soaked almonds (6-8)")
            fat_sources.insert(1, "Walnuts (3-4)")
        fats_kya = ", ".join(fat_sources[:5])

        # Carbs curated sources honoring gluten allergies
        carb_sources = ["Brown basmati rice", "Sweet potatoes", "Bananas", "Poha", "Tender coconut water", "Boiled potatoes"]
        if not has_allergy_gluten:
            carb_sources.insert(0, "Rolled oats")
            carb_sources.insert(1, "Multigrain rotis")
        carbs_kya = ", ".join(carb_sources[:5])

        macros_rationale = {
            "calories": {
                "name": "Daily Caloric Energy",
                "kitna": f"Match Bowling Day: {target_cals + 350} kcal | Training Day: {target_cals} kcal | Recovery Day: {max(2200, target_cals - 450)} kcal",
                "kyun": (
                    f"Fast bowling ek high-demand glycolytic action hai. Ek heavy spell (8-10 overs) + fielding me ~950-1200 kcal burn hote hain. "
                    f"Caloric deficit me bowler ki pace 4-7 km/h drop hoti hai aur body eccentric hamstring tissues ko fuel ke liye break down karne lagti hai."
                ),
                "kya_khana": carbs_kya
            },
            "protein": {
                "name": "Protein & Amino Synthesis",
                "kitna": f"{base_protein}g / day (2.2g per kg bodyweight for {user_weight} kg athlete). Per meal: 30-35g with 3.0g Leucine threshold.",
                "kyun": (
                    f"Front-foot contact par 8-9x bodyweight ground reaction force front knee patellar tendon aur hamstrings me microscopic micro-tears produce karti hai. "
                    f"High-leucine protein synthesis tendon collagen remodeling aur muscle myofibrillar recovery ke liye mandatory hai."
                ),
                "kya_khana": protein_kya
            },
            "carbohydrates": {
                "name": "Carbohydrates & Glycogen Primer",
                "kitna": f"~{round((target_cals * 0.50) / 4)}g / day (4.5 - 5.5g per kg). Pre-spell window: 40-50g fast-acting carbs.",
                "kyun": (
                    f"{max_speed} km/h bowling velocity ke run-up aur delivery stride strictly phosphagen (ATP-CP) aur fast glycolysis par depend karte hain. "
                    f"Glycogen depletion se 5th-6th over me arm speed drop hoti hai aur bowler kinetic chain twist karke lumbar spine par excess load dalne lagta hai."
                ),
                "kya_khana": "Bananas with raw honey, boiled sweet potatoes, rolled oats, raisins, tender coconut water, brown basmati rice."
            },
            "fats_and_joint_care": {
                "name": "Healthy Fats, Spine & Joint Care",
                "kitna": f"~{round((target_cals * 0.25) / 9)}g / day (healthy lipids) + {(user_weight * 0.055):.1f}L electrolyte hydration.",
                "kyun": (
                    f"L4-L5 lumbar vertebrae ki disc hydration aur facet joint cushioning ke liye Omega-3 fatty acids aur anti-inflammatory lipids zaroori hain. "
                    f"Electrolytes (Sodium & Potassium) repetitive delivery strides me cramping aur spinal fatigue prevent karte hain."
                ),
                "kya_khana": fats_kya
            }
        }

        # -------------------------------------------------------------
        # 4. DAY-WISE SCHEDULE GENERATION (LLM + Calculated Workload & Nutrition)
        # -------------------------------------------------------------
        schedule = self._generate_llm_schedule(
            payload=payload,
            calculated_metrics={
                "bmr": bmr,
                "base_tdee": base_tdee,
                "target_cals": target_cals,
                "base_protein": base_protein,
                "estimated_weekly_overs": estimated_weekly_overs,
                "overs_ceiling": overs_ceiling,
                "spell_ceiling": spell_ceiling,
                "acwr_score": acwr_score,
                "acwr_status": acwr_status,
                "weekly_bowling_days": weekly_bowling_days,
                "ground_days": ground_days,
                "gym_days": gym_days,
                "max_overs_per_spell": max_overs_per_spell,
                "avg_overs_per_session": avg_overs_per_session,
                "max_speed": max_speed,
                "injury_status": injury_status,
                "session_gap": session_gap,
                "athlete_status": athlete_status,
                "user_weight": user_weight,
                "user_height": user_height,
                "user_age": user_age,
                "user_sex": user_sex,
                "goal": goal,
                "food_pref": food_pref,
                "is_veg": is_veg,
                "selected_allergies": selected_allergies,
                "weekly_budget": weekly_budget,
                "strength_track": strength_track,
                "deadlift_kg": deadlift_kg,
                "est_deadlift_1rm": est_deadlift_1rm,
                "does_deadlifts": does_deadlifts,
                "has_gym": has_gym,
                "bowling_arm": bowling_arm,
                "bowling_action": bowling_action,
                "runup_length": runup_length,
            },
            llm_client=None
        )

        calculated_bmr = self.calculate_bmr(user_weight, user_height, user_age, user_sex)

        ai_assessment = self.generate_ai_coaching_assessment(
            athlete_profile={
                "age": user_age,
                "weight": user_weight,
                "height": user_height,
                "sex": user_sex,
                "goal": goal,
                "foodPref": food_pref,
                "athlete_status": athlete_status,
                "max_speed": max_speed,
                "injury_status": injury_status,
                "weekly_bowling_days": weekly_bowling_days,
                "max_overs_per_spell": max_overs_per_spell,
                "avg_overs_per_session": avg_overs_per_session,
            },
            nutrition_metrics={
                "bmr": calculated_bmr,
                "tdee": base_tdee,
                "target_calories": target_cals,
                "protein_g": base_protein,
                "carbs_g": round((target_cals * 0.50) / 4),
                "fats_g": round((target_cals * 0.25) / 9),
            },
            workload_metrics={
                "acwr_score": acwr_score,
                "acwr_status": acwr_status,
                "total_weekly_overs": estimated_weekly_overs,
                "gym_days": gym_days,
                "ground_days": ground_days,
                "weekly_bowling_days": weekly_bowling_days,
                "max_overs_per_spell": max_overs_per_spell,
                "avg_overs_per_session": avg_overs_per_session,
                "max_speed": max_speed,
                "injury_status": injury_status,
                "overs_ceiling": overs_ceiling,
            }
        )

        # Staged Action Targets (Phased Development Roadmap)
        if does_deadlifts:
            phase_1_targets = [
                f"Gain +3 to 4 kg Lean Mass (Reach target BMI 19.5+) to absorb 8-9x BW front foot contact forces" if is_underweight else "Solidify Posterior Chain & Lumbar Core Stability for front knee brace firmness",
                f"Progress Deadlift from {deadlift_kg}kg to {round(user_weight * 1.5)}kg (1.5x Bodyweight Target)",
                f"Lock daily water intake to minimum {max(3.8, round(user_weight * 0.055, 1))}L with electrolytes to prevent disc thinning & cramps",
                f"Cap weekly bowling strictly at safe ceiling ({overs_ceiling} overs/wk) with 48h spell spacing"
            ]
        else:
            phase_1_targets = [
                f"Gain +3 to 4 kg Lean Mass (Reach target BMI 19.5+) with high-protein calisthenic armor" if is_underweight else "Master Bodyweight Posterior Chain (Nordic Hamstring Drops & Glute Bridges)",
                f"Progress Push-ups to 30+ reps & Single-Leg Wall Sits (60s hold) for front-knee lock",
                f"Lock daily water intake to minimum {max(3.8, round(user_weight * 0.055, 1))}L with electrolytes to prevent disc thinning & cramps",
                f"Cap weekly bowling strictly at safe ceiling ({overs_ceiling} overs/wk) with 48h spell spacing"
            ]

        phase_2_targets = [
            f"Lead-Leg Eccentric Stiffness: {'Bulgarian Split Squats (3 sets × 8 reps with ' + str(round(user_weight * 0.25)) + 'kg dumbbells)' if has_gym else 'Bodyweight Bulgarian Split Squats (3 sets × 15 reps each leg)'}",
            "Isometric Front-Knee Stiffness: 3 sets × 60s Wall Sits with knees at 90° post-bowling",
            "Lumbar Anti-Rotation: Core Paloff presses (or Ground Hollow Body holds) + 60s planks",
            "Short run-up spot bowling (4 overs) focusing on 165°-175° rigid front-knee plant"
        ]
        phase_3_targets = [
            "Kinetic Energy Whip: Rotational Medicine Ball Slams (or explosive ground plyos) for torso unwinding",
            f"Full Run-Up Velocity Conversion: Unleash peak spells targeting > {max_speed} km/h",
            "Match Workload Regulation: Keep ACWR locked in sweet spot (0.90 – 1.25) during tournament spells"
        ]
        staged_targets = {
            "current_focus": "Phase 1: Armor, Mass & Foundation" if (is_underweight or (does_deadlifts and is_strength_deficient) or is_calisthenics_deficient or is_dehydrated or is_overs_overloaded) else "Phase 2: Lead-Leg Stiffness",
            "phase_1": {
                "title": "Phase 1: Foundation, Mass Armor & Base Strength (Weeks 1–4)",
                "summary": "Build the muscular armor and posterior chain foundation required to absorb 8-9x bodyweight delivery strides without spinal injury.",
                "targets": phase_1_targets
            },
            "phase_2": {
                "title": "Phase 2: Unilateral Lead-Leg Stiffness & Knee Brace (Weeks 5–8)",
                "summary": "Develop the eccentric brake capacity on the front leg to convert horizontal run-up momentum into pure vertical ball velocity.",
                "targets": phase_2_targets
            },
            "phase_3": {
                "title": "Phase 3: Kinetic Transfer & Peak Match Velocity (Weeks 9–12)",
                "summary": "Translate strength gains into explosive bowling speed while maintaining strict workload discipline to prevent overtraining.",
                "targets": phase_3_targets
            }
        }

        strength_assessment = {
            "training_track": "Gym & Weight Training" if has_gym else "Ground & Home Calisthenics (No Gym)",
            "has_gym": has_gym,
            "does_deadlifts": does_deadlifts,
            "deadlift_kg": deadlift_kg if does_deadlifts else 0,
            "deadlift_reps": deadlift_reps if does_deadlifts else 0,
            "estimated_1rm_kg": est_deadlift_1rm if does_deadlifts else 0,
            "strength_ratio": strength_ratio if does_deadlifts else 0,
            "split_squat_kg": split_squat_kg if has_gym else 0,
            "split_squat_reps": split_squat_reps,
            "bodyweight_squats_reps": bodyweight_squats_reps,
            "pushups_reps": pushups_reps,
            "pullups_reps": pullups_reps,
            "plank_seconds": plank_seconds,
            "status": (
                ("Needs Muscular Armor & Base Strength" if strength_ratio < 1.40 else ("Intermediate Athletic Foundation" if strength_ratio < 1.75 else "Elite Fast Bowler Posterior Chain"))
                if does_deadlifts else
                ("Needs Calisthenic Armor & Base" if is_calisthenics_deficient else "Strong Ground & Calisthenic Durability Foundation")
            )
        }

        return {
            "id": f"cp_{int(datetime.now().timestamp() * 1000)}",
            "status": "success",
            "title": f"Fast Bowler Workload & Athletic Routine ({'Student Bowler' if athlete_status == 'student' else 'Full-Time Pro'} Profile)",
            "athlete_status": athlete_status,
            "bmi": bmi,
            "bmi_category": "underweight" if is_underweight else ("overweight" if is_overweight else "normal"),
            "strength_assessment": strength_assessment,
            "staged_targets": staged_targets,
            "workload_summary": {
                "gym_days": gym_days,
                "ground_days": ground_days,
                "weekly_bowling_days": weekly_bowling_days,
                "max_overs_per_spell": max_overs_per_spell,
                "avg_overs_per_session": avg_overs_per_session,
                "max_speed": max_speed,
                "injury_status": injury_status,
                "bowling_arm": bowling_arm,
                "bowling_action": bowling_action,
                "runup_length": runup_length,
                "sleep_hours": sleep_hours,
                "water_intake_liters": water_intake_liters,
                "supplement_preference": supplement_preference,
                "meal_frequency": meal_frequency,
                "max_overs_days": max_overs_days,
                "session_gap": session_gap,
                "total_weekly_overs": estimated_weekly_overs,
                "overs_ceiling": overs_ceiling,
                "spell_ceiling": spell_ceiling,
                "acwr_score": acwr_score,
                "acwr_status": acwr_status,
                "is_overloaded": is_overs_overloaded or is_spell_overloaded or is_back_to_back or is_injury_speed_clash or is_mixed_action or is_sleep_deprived or is_dehydrated or is_underweight,
                "fatigue_index": "High Strain (Prioritize Sleep & Hydration)" if (total_sessions > 7 or is_overs_overloaded or is_sleep_deprived or is_dehydrated) else "Well Balanced",
                "ecb_guideline_check": ecb_check,
            },
            "nutrition_summary": {
                "bmr": calculated_bmr,
                "base_tdee": base_tdee,
                "training_day_calories": target_cals,
                "max_overs_calories": target_cals + 350,
                "recovery_day_calories": max(2200, target_cals - 450),
                "daily_protein_target_g": base_protein,
                "daily_carbs_target_g": round((target_cals * 0.50) / 4),
                "daily_fats_target_g": round((target_cals * 0.25) / 9),
                "estimated_weekly_budget_inr": weekly_budget,
                "hydration_target": f"{(user_weight * 0.05):.1f} - {(user_weight * 0.06):.1f} Liters/day (electrolytes mandatory during bowling)",
            },
            "overload_warnings": overload_warnings,
            "current_mistakes_and_gaps": current_mistakes_and_gaps,
            "macros_rationale": macros_rationale,
            "athlete_profile": {
                "age": user_age,
                "weight_kg": user_weight,
                "height_cm": user_height,
                "sex": user_sex,
                "goal": goal,
                "bmi": bmi,
                "deadlift_kg": deadlift_kg,
                "deadlift_reps": deadlift_reps,
                "strength_ratio": strength_ratio,
                "food_preference": food_pref,
                "weekly_budget": weekly_budget,
                "weekly_bowling_days": weekly_bowling_days,
                "max_overs_per_spell": max_overs_per_spell,
                "avg_overs_per_session": avg_overs_per_session,
                "max_speed": max_speed,
                "injury_status": injury_status,
                "bowling_arm": bowling_arm,
                "bowling_action": bowling_action,
                "runup_length": runup_length,
                "sleep_hours": sleep_hours,
                "water_intake_liters": water_intake_liters,
                "supplement_preference": supplement_preference,
                "meal_frequency": meal_frequency,
                "protein_g_per_kg": 2.2,
                "daily_protein_target_g": base_protein,
                "daily_carbs_target_g": round((target_cals * 0.50) / 4),
                "daily_fats_target_g": round((target_cals * 0.25) / 9),
                "bmr": calculated_bmr,
                "base_tdee": base_tdee,
                "bowling_calories": target_cals + 350,
                "training_calories": target_cals,
                "recovery_calories": max(2200, target_cals - 450),
                "hydration_liters": round(user_weight * 0.055, 1),
                "max_overs_ceiling": overs_ceiling,
                "weekly_overs": estimated_weekly_overs,
                "total_sessions": total_sessions,
                "acwr_score": acwr_score,
                "acwr_status": acwr_status,
                "acwr_zone": "sweet_spot" if acwr_score <= 1.3 else ("elevated" if acwr_score <= 1.40 else "danger_zone"),
            },
            "schedule": schedule,
            "ai_coaching_assessment": ai_assessment,
            "ai_assessment": ai_assessment,
        }

    def generate_workout_plan(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Return the workout schedule in the shape used by the workout page."""
        combined_plan = self.generate_combined_plan(payload)
        weekly_schedule = []

        for day in combined_plan.get("schedule", []):
            details = []
            ground_session = day.get("workload", {}).get("ground_session")
            gym_session = day.get("workload", {}).get("gym_session")

            if ground_session:
                drills = ground_session.get("drills", [])
                if isinstance(drills, list):
                    details.extend(drills)
                elif isinstance(drills, str):
                    details.append(drills)
            if gym_session:
                exercises = gym_session.get("exercises", [])
                if isinstance(exercises, list):
                    for exercise in exercises:
                        if isinstance(exercise, dict):
                            details.append(f'{exercise.get("name", "Exercise")} {exercise.get("sets", 3)}x{exercise.get("reps", "8-10 reps")}')
                        elif isinstance(exercise, str):
                            details.append(exercise)
            if not details:
                details.append("Mobility, spinal decompression, hydration, and recovery work")

            weekly_schedule.append({
                "day": str(day.get("day", "DAY")).upper(),
                "title": str(day.get("title", "Training Session")),
                "details": details,
            })

        return {
            "id": f'wp_{int(datetime.now().timestamp() * 1000)}',
            "title": f"Fast Bowler 7-Day Performance Workplan ({combined_plan.get('workload_summary', {}).get('max_speed', 132)} km/h Focus)",
            "focus": f"Speed, ACWR Workload ({combined_plan.get('workload_summary', {}).get('acwr_score', 1.18)}), and lumbar spine protection",
            "weeklySchedule": weekly_schedule,
            "days_count": len(weekly_schedule),
            "workload_summary": combined_plan.get("workload_summary", {}),
            "strength_assessment": combined_plan.get("strength_assessment", {}),
            "ai_coaching_assessment": combined_plan.get("ai_coaching_assessment"),
            "ai_assessment": combined_plan.get("ai_assessment"),
        }

    def _generate_llm_schedule(
        self,
        payload: Dict[str, Any],
        calculated_metrics: Dict[str, Any],
        llm_client: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Calls LLM to generate the synchronized 7-day cricket workload & nutrition
        schedule matching the exact calculated training days, overs, speed,
        and caloric targets. Falls back to deterministic engine if offline.
        """
        client = llm_client or LLMClient()
        cm = calculated_metrics

        system_prompt = (
            "You are PacePro AI's Lead Fast Bowling Strength & Conditioning Coach and Head Sports Scientist.\n"
            "You are provided with pre-calculated, verified sports-science metrics for a cricket fast bowler:\n"
            f"- Athletic Profile: {cm.get('user_age')} yrs, {cm.get('user_weight')} kg, {cm.get('user_height')} cm, Goal: '{cm.get('goal')}', Status: '{cm.get('athlete_status')}'\n"
            f"- Bowling Pace: {cm.get('max_speed')} km/h ({cm.get('bowling_arm')}-arm, {cm.get('bowling_action')} action)\n"
            f"- Workload Constraints: Exactly {cm.get('ground_days')} Ground sessions, {cm.get('gym_days')} Gym sessions, {cm.get('weekly_bowling_days')} Bowling Spells (Max spell: {cm.get('max_overs_per_spell')} overs)\n"
            f"- ACWR Workload: {cm.get('acwr_score')} ({cm.get('acwr_status')}), Safe Weekly Ceiling: {cm.get('overs_ceiling')} overs\n"
            f"- Calculated Calories: Bowling Days ({cm.get('target_cals') + 350} kcal), Training Days ({cm.get('target_cals')} kcal), Recovery Days ({max(2200, cm.get('target_cals') - 450)} kcal)\n"
            f"- Protein Target: {cm.get('base_protein')}g daily\n"
            f"- Food Preference: '{cm.get('food_pref')}' | Allergies to avoid: {json.dumps(cm.get('selected_allergies', []))}\n\n"
            "CRITICAL RULES:\n"
            "1. Generate exactly 7 synchronized days (Monday through Sunday).\n"
            "2. Strictly respect the ground and gym session counts and ensure bowling match spells are spaced by ~48h.\n"
            "3. Ground sessions must have specific cricket drills (e.g. Marker sprints, 5 overs match pace, yorkers). Gym sessions must have specific fast-bowling exercises (e.g. Trap Bar Deadlift 3x5, Bulgarian Split Squat 3x8, Pallof Press 3x12) with sets and reps.\n"
            "4. Keep meal items clear, crisp, and realistic for Indian cricketers.\n"
            "5. Respond ONLY with a valid JSON array of 7 day objects matching this exact schema:\n"
            "[\n"
            "  {\n"
            '    "day": "Monday",\n'
            '    "day_type": "max_overs_bowling",\n'
            '    "badge": "Max Overs Spell",\n'
            '    "title": "High-Intensity Match Spell (8 Overs @ 132 km/h)",\n'
            '    "workload": {\n'
            '      "ground_session": {"overs": 8, "intensity": "100% Match Pace", "focus": "Yorkers & Brace Lock", "drills": ["Run-up cadence marker sprints (4 reps)", "Good-length line & length targets (5 overs)", "Death-overs wide yorkers (3 overs)", "Thoracic decompression dead hangs"]},\n'
            '      "gym_session": null\n'
            '    },\n'
            '    "nutrition": {\n'
            f'      "calories": {cm.get("target_cals") + 350},\n'
            f'      "protein_g": {cm.get("base_protein")},\n'
            f'      "carbs_g": {round((cm.get("target_cals") * 0.50) / 4)},\n'
            f'      "fats_g": {round((cm.get("target_cals") * 0.25) / 9)},\n'
            '      "focus": "High glycogen replenishment & tendon collagen repair",\n'
            '      "workload_adaptation": "Pre-spell fast carbs 45m before and post-session whey/dahi within 30m",\n'
            '      "meal_options": {\n'
            '        "option_a": {"name": "Option A: Elite Performance Fuel", "meals": [{"time": "07:30 AM", "name": "Breakfast", "items": "100g Rolled Oats with Whey, 2 Boiled Eggs/Paneer, Banana"}, {"time": "01:00 PM", "name": "Lunch", "items": "Grilled Chicken/Paneer, Brown Rice, Dal, Curd"}, {"time": "08:30 PM", "name": "Dinner", "items": "3 Rotis, Yellow Dal, Soya/Fish, Spinach Salad"}]},\n'
            '        "option_b": {"name": "Option B: Budget Indian Fuel", "meals": [{"time": "07:30 AM", "name": "Breakfast", "items": "Sattu Drink (50g), 4 Boiled Eggs/Sprouts, 2 Bananas"}, {"time": "01:00 PM", "name": "Lunch", "items": "Rajma/Chole Chawal, Greek Dahi, Cucumber Salad"}, {"time": "08:30 PM", "name": "Dinner", "items": "3 Multigrain Rotis, Moong Dal Khichdi, Paneer Bhurji"}]}'
            '      }\n'
            '    },\n'
            '    "student_schedule": "06:15 AM - 08:00 AM Ground | 09:00 AM - 04:00 PM Classes | 05:00 PM Recovery",\n'
            '    "fulltime_schedule": "07:30 AM - 10:00 AM Bowling Spell | 11:30 AM Contrast Shower & Lunch | 04:00 PM Mobility",\n'
            '    "session_gap_advisory": "Single high-velocity session today. Maintain 48h before next heavy bowling spell."\n'
            "  }\n"
            "]"
        )

        user_prompt = (
            f"Generate the complete 7-Day Fast Bowler Schedule (Monday to Sunday) for {cm.get('user_weight')}kg bowler targeting {cm.get('goal')}.\n"
            f"Ground Days: {cm.get('ground_days')} | Gym Days: {cm.get('gym_days')} | Bowling Spells: {cm.get('weekly_bowling_days')} | Speed: {cm.get('max_speed')} km/h.\n"
            f"Diet: {cm.get('food_pref')} | Target Calories: {cm.get('target_cals')} kcal | Protein: {cm.get('base_protein')}g."
        )

        try:
            parsed = client.generate_json(system_prompt, user_prompt)
            if isinstance(parsed, dict) and "schedule" in parsed:
                parsed = parsed["schedule"]
            if isinstance(parsed, list) and len(parsed) == 7:
                cleaned_schedule = []
                for d in parsed:
                    if isinstance(d, dict) and "day" in d:
                        cleaned_schedule.append({
                            "day": str(d.get("day", "Day")),
                            "day_type": str(d.get("day_type", "ground_skill")),
                            "badge": str(d.get("badge", "Training")),
                            "title": str(d.get("title", "Bowling Training")),
                            "workload": d.get("workload", {}),
                            "nutrition": d.get("nutrition", {}),
                            "student_schedule": str(d.get("student_schedule", "Study & Ground Sessions")),
                            "fulltime_schedule": str(d.get("fulltime_schedule", "Pro Athlete Schedule")),
                            "active_schedule": str(d.get("student_schedule" if cm.get("athlete_status") == "student" else "fulltime_schedule", "")),
                            "session_gap_advisory": str(d.get("session_gap_advisory", cm.get("session_gap", "7-8h gap between sessions"))),
                        })
                if len(cleaned_schedule) == 7:
                    return cleaned_schedule
        except Exception as e:
            logger.warning(f"[PlannerService] LLM 7-day schedule generation failed or offline: {e}")

        # Fallback to sports-science engine schedule
        return self._build_fallback_schedule(calculated_metrics)

    def _build_fallback_schedule(self, cm: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Deterministic, sports-science calculated 7-day synchronized schedule
        matching exact ground/gym days, max-overs spells, and nutritional targets.
        """
        days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        schedule = []
        rem_bowling_days = cm.get("weekly_bowling_days", 3)
        rem_ground = cm.get("ground_days", 4)
        rem_gym = cm.get("gym_days", 3)
        
        target_cals = cm.get("target_cals", 2800)
        base_protein = cm.get("base_protein", 150)
        max_overs_per_spell = cm.get("max_overs_per_spell", 8)
        avg_overs_per_session = cm.get("avg_overs_per_session", 5)
        max_speed = cm.get("max_speed", 132.0)
        session_gap = cm.get("session_gap", "7-8h gap between sessions")
        athlete_status = cm.get("athlete_status", "student")
        is_veg = cm.get("is_veg", False)

        for idx, day_name in enumerate(days_of_week):
            is_match_spell = False
            has_ground = False
            has_gym = False

            if rem_bowling_days > 0 and (idx == 0 or idx == 3 or (idx == 5 and rem_bowling_days >= 3)):
                is_match_spell = (idx == 0 or (idx == 3 and rem_bowling_days >= 2))
                has_ground = True
                rem_bowling_days -= 1
                rem_ground = max(0, rem_ground - 1)
            elif rem_ground > 0 and (idx in (0, 2, 4)):
                has_ground = True
                rem_ground -= 1

            if rem_gym > 0 and (idx in (1, 3, 5) or (idx == 0 and rem_gym >= 4)):
                has_gym = True
                rem_gym -= 1

            if is_match_spell:
                day_type = "max_overs_bowling"
                badge = "Max Overs Spell"
                title = f"High-Intensity Match Spell ({max_overs_per_spell} Overs @ {max_speed} km/h)"
                ground_session = {
                    "overs": max_overs_per_spell,
                    "intensity": f"100% Match Pace ({max_speed} km/h)",
                    "focus": "Full-speed match spell with front knee brace firmness and yorker target execution",
                    "drills": [
                        "Dynamic run-up markers × 4 acceleration build-ups",
                        f"Target bowling on 6-8 meter good length ({max(3, max_overs_per_spell - 3)} overs)",
                        f"Death overs yorkers and wide yorkers ({min(3, max_overs_per_spell)} overs)",
                        "Post-spell 3x45s dead hangs for lumbar decompression"
                    ],
                }
                gym_session = None
                nutrition_cals = target_cals + 350
                day_protein = base_protein
                day_carbs = round((nutrition_cals * 0.55) / 4)
                day_fats = round((nutrition_cals * 0.22) / 9)
                workload_adaptation = (
                    f"High glycogen pre-load and anti-inflammatory recovery calibrated for {max_overs_per_spell} overs "
                    f"at match pace ({max_speed} km/h) and heavy front-knee brace ground reaction forces."
                )
                option_a_meals = [
                    {"time": "06:00 AM", "name": "Pre-Spell Glycogen Primer", "items": "2 ripe bananas with raw honey, 1 multigrain toast, 350ml electrolyte hydration water"},
                    {"time": "08:30 AM", "name": "Rapid Post-Spell Recovery", "items": "1 scoop whey/plant isolate in tender coconut water, 1 large boiled sweet potato"},
                    {"time": "10:30 AM", "name": "Athletic Power Breakfast", "items": "Paneer bhurji (150g) with 3 chapatis and seasonal fruit bowl" if is_veg else "3 egg whites + 1 whole egg omelette with 2 multigrain toasts, fruit bowl"},
                    {"time": "02:00 PM", "name": "Tissue Repair Lunch", "items": "200g soya chunk curry with 200g brown basmati rice, steamed spinach, probiotic curd" if is_veg else "200g grilled chicken breast with 200g brown basmati rice, steamed spinach, curd"},
                    {"time": "05:30 PM", "name": "Joint De-Stress Recovery Snack", "items": "Roasted Bengal gram (chana, 60g), soaked pumpkin seeds, green tea"},
                    {"time": "08:30 PM", "name": "Spine & Lumbar Healing Dinner", "items": "3 multigrain rotis with yellow dal tadka, sautéed paneer, mixed salad, warm golden turmeric milk" if is_veg else "3 multigrain rotis with grilled fish or chicken, dal tadka, salad, turmeric milk"}
                ]
                option_b_meals = [
                    {"time": "06:00 AM", "name": "Budget Pre-Spell Quick Fuel", "items": "2 large bananas with jaggery (gud), pinch of black salt in lemon water"},
                    {"time": "08:30 AM", "name": "Budget Post-Spell Recovery", "items": "1 glass fresh sattu drink (40g chana sattu) with lemon, 1 boiled potato with rock salt"},
                    {"time": "10:30 AM", "name": "High-Value Breakfast", "items": "Poha with roasted peanuts & boiled sprouts (100g), 1 glass milk" if is_veg else "3 boiled eggs with 2 plain rotis and sliced tomato-cucumber salad"},
                    {"time": "02:00 PM", "name": "Budget High-Protein Lunch", "items": "70g boiled soya chunks in rich tomato-onion curry with 250g steamed rice and yellow dal"},
                    {"time": "05:30 PM", "name": "Affordable Evening Fuel", "items": "Roasted makhana or roasted chana (50g) with hot jaggery tea"},
                    {"time": "08:30 PM", "name": "Overnight Repair Dinner", "items": "Moong dal khichdi with 1 tbsp pure desi ghee, curd, and cucumber salad"}
                ]
                student_sched = f"06:15 AM - 08:00 AM: Ground {max_overs_per_spell} Overs Match Spell | 09:00 AM - 04:00 PM: College | Evening: Foam roll & Ice"
                athlete_sched = f"07:30 AM - 10:00 AM: High-Velocity Match Spell ({max_speed} km/h) | 11:30 AM: Physio & Cold Plunge | 04:00 PM: Video Analysis"

            elif has_ground and has_gym:
                day_type = "dual_session"
                badge = "Dual Session"
                title = f"Ground Technical Drills ({avg_overs_per_session} Overs) + Gym Power"
                ground_session = {
                    "overs": avg_overs_per_session,
                    "intensity": "85% Technical Rhythm",
                    "focus": "Approach cadence, steady seam presentation, and release angle consistency",
                    "drills": ["Crease alignment steps", f"Spot bowling ({avg_overs_per_session} overs)", "Follow-through hold for balance"],
                }
                gym_session = {
                    "status": "Lower Body Power & Kinetic Chain",
                    "duration": "60 mins",
                    "focus": "Trap bar deadlifts, single-leg landing stiffness & rotational core",
                    "exercises": [
                        {"name": "Trap Bar Deadlift", "sets": 4, "reps": "5 reps @ 75% 1RM"},
                        {"name": "Single-Leg Box Jumps (Stick Landing)", "sets": 3, "reps": "5 each leg"},
                        {"name": "Pallof Press Core Hold", "sets": 3, "reps": "30s each side"}
                    ],
                }
                nutrition_cals = target_cals + 200
                day_protein = base_protein + 10
                day_carbs = round((nutrition_cals * 0.50) / 4)
                day_fats = round((nutrition_cals * 0.25) / 9)
                workload_adaptation = (
                    "Split energy pulses: rapid complex carbs before morning ground drills, high-protein synthesis support "
                    "between sessions and after evening gym strength training."
                )
                option_a_meals = [
                    {"time": "06:00 AM", "name": "Ground Session Primer", "items": "1 banana with 1 tbsp peanut/almond butter, black coffee, 300ml electrolyte water"},
                    {"time": "08:30 AM", "name": "Post-Ground Hearty Breakfast", "items": "Rolled oats (100g) with protein powder, chia seeds, sliced apple and soaked nuts" if is_veg else "4 boiled eggs (2 yolks), rolled oats porridge with sliced apple and nuts"},
                    {"time": "01:00 PM", "name": "Midday Glycogen & Protein Plate", "items": "200g paneer curry with 3 chapatis, cucumber raita, sprout salad" if is_veg else "200g chicken breast curry with 3 chapatis, cucumber raita, sprout salad"},
                    {"time": "04:45 PM", "name": "Pre-Gym Posterior Chain Primer", "items": "Handful of raisins, 8 soaked almonds, 1 slice multigrain toast, green tea"},
                    {"time": "07:00 PM", "name": "Post-Gym Protein Pulse", "items": "1 scoop whey isolate in water or plant milk"},
                    {"time": "08:45 PM", "name": "Restorative Dinner", "items": "3 multigrain rotis, thick yellow moong dal, paneer bhurji, beetroot salad" if is_veg else "3 multigrain rotis, thick moong dal, grilled chicken/fish, beetroot salad"}
                ]
                option_b_meals = [
                    {"time": "06:00 AM", "name": "Budget Morning Wake-up", "items": "Warm jaggery water with lemon, 1 banana, handful of roasted chana"},
                    {"time": "08:30 AM", "name": "Post-Ground Fuel Breakfast", "items": "Besan & moong dal chilla (3 chillas) stuffed with grated paneer/potatoes, mint chutney"},
                    {"time": "01:00 PM", "name": "Budget Power Lunch", "items": "Steamed rice with thick rajma or chole curry, cabbage salad, curd"},
                    {"time": "04:45 PM", "name": "Pre-Gym Energy Boost", "items": "1 boiled sweet potato or 2 bananas with black coffee"},
                    {"time": "07:00 PM", "name": "Post-Gym Low-Cost Protein", "items": "40g Sattu drink with water and rock salt, or 3 egg whites"},
                    {"time": "08:45 PM", "name": "Restorative Budget Dinner", "items": "3 chapatis with soya chunk curry, mixed green sabzi, glass of warm milk"}
                ]
                student_sched = "06:30 AM - 08:00 AM: Ground Drills | 09:00 AM - 04:30 PM: Classes | 05:30 PM - 06:45 PM: Strength Gym"
                athlete_sched = "08:00 AM - 10:00 AM: Technical Ground | 01:00 PM: Nutrition & Rest | 04:30 PM - 06:00 PM: Explosive Gym"

            elif has_ground:
                day_type = "ground_skill"
                badge = "Ground Technical"
                title = f"Run-up Cadence & Spot Bowling ({avg_overs_per_session} Overs)"
                ground_session = {
                    "overs": avg_overs_per_session,
                    "intensity": "75% Spot Bowling",
                    "focus": "Good length consistency, wrist snap, and front-knee brace firmness",
                    "drills": ["Cone targets on 6-8 meter mark", "Approach acceleration sprints × 6", "Wrist release hold drills"],
                }
                gym_session = None
                nutrition_cals = target_cals
                day_protein = base_protein
                day_carbs = round((nutrition_cals * 0.50) / 4)
                day_fats = round((nutrition_cals * 0.25) / 9)
                workload_adaptation = (
                    "Steady hydration and moderate complex carbs to maintain run-up cadence and release angle "
                    "sharpness without digestive heaviness."
                )
                option_a_meals = [
                    {"time": "06:30 AM", "name": "Light Bowling Fuel", "items": "Warm lemon honey water, 1 banana, 6 soaked almonds"},
                    {"time": "08:30 AM", "name": "Technical Recovery Breakfast", "items": "Moong dal chilla (3 chillas) with paneer filling, mint chutney, tender coconut water" if is_veg else "Egg scramble (3 eggs) with 2 multigrain rotis, sliced kiwi, coconut water"},
                    {"time": "01:30 PM", "name": "Balanced Cricket Lunch", "items": "Brown basmati rice with light dal, paneer bhurji, carrot and cucumber salad" if is_veg else "Brown basmati rice with chicken curry, boiled egg, salad"},
                    {"time": "05:30 PM", "name": "Afternoon Refuel", "items": "Sprouted moong chaat with chopped tomatoes, lemon juice, and pumpkin seeds"},
                    {"time": "08:30 PM", "name": "Light Evening Meal", "items": "Vegetable paneer pulao with curd, 2 rotis, chamomile tea" if is_veg else "Grilled fish or chicken breast with 2 rotis, soup, chamomile tea"}
                ]
                option_b_meals = [
                    {"time": "06:30 AM", "name": "Budget Pre-Ground Fuel", "items": "1 banana, handful of soaked black chana, warm water"},
                    {"time": "08:30 AM", "name": "Wholesome Breakfast", "items": "Vegetable upma with roasted peanuts/sprouts, 1 cup curd"},
                    {"time": "01:30 PM", "name": "Everyday Athletic Lunch", "items": "3 rotis, bowl of yellow dal, aloo-palak sabzi, cucumber salad"},
                    {"time": "05:30 PM", "name": "Evening Snack", "items": "Boiled corn or roasted chana chaat with lemon and onion"},
                    {"time": "08:30 PM", "name": "Light Dinner", "items": "Moong dal khichdi with roasted papad, 1 cup curd, warm turmeric milk"}
                ]
                student_sched = "06:30 AM - 08:00 AM: Ground | College / Study day | Evening stretching & dead hangs"
                athlete_sched = "08:30 AM - 11:00 AM: Spot Bowling & Seam Release Clinic | Afternoon S&C review"

            elif has_gym:
                day_type = "gym_strength"
                badge = "Gym Strength"
                title = "Kinetic Chain & Lumbar Core Resilience"
                ground_session = None
                gym_session = {
                    "status": "Strength & Core",
                    "duration": "60 mins",
                    "focus": "Lumbar protection, rotational anti-flexion & posterior chain",
                    "exercises": [
                        {"name": "Pallof Press Hold", "sets": 3, "reps": "30s each side"},
                        {"name": "Bulgarian Split Squats", "sets": 3, "reps": "8 each leg"},
                        {"name": "Nordic Hamstring Curls (Assisted)", "sets": 3, "reps": "5 reps"}
                    ],
                }
                nutrition_cals = target_cals
                day_protein = base_protein + 12
                day_carbs = round((nutrition_cals * 0.45) / 4)
                day_fats = round((nutrition_cals * 0.28) / 9)
                workload_adaptation = (
                    "Hyper-protein synthesis target for lumbar spine core stabilizers and eccentric hamstring "
                    "tendon resilience."
                )
                option_a_meals = [
                    {"time": "07:30 AM", "name": "Anabolic Foundation Breakfast", "items": "100g oats with peanut butter, chia seeds, and 1 scoop plant/whey protein powder" if is_veg else "3 whole eggs scrambled, 2 slices multigrain toast, glass of milk"},
                    {"time": "11:00 AM", "name": "Mid-Morning Muscle Fuel", "items": "Greek yogurt with crushed walnuts, berries, and raw honey"},
                    {"time": "01:30 PM", "name": "High-Protein Power Lunch", "items": "High-protein soya chunks and paneer curry, 3 chapatis, dal, mixed greens" if is_veg else "220g chicken breast curry with 3 chapatis, dal, mixed greens"},
                    {"time": "04:30 PM", "name": "Pre-Workout Strength Primer", "items": "1 banana with 1 tbsp peanut butter, black coffee"},
                    {"time": "06:45 PM", "name": "Post-Workout Whey Refuel", "items": "1 scoop whey isolate with water, 1 fresh seasonal fruit"},
                    {"time": "08:45 PM", "name": "Overnight Recovery Dinner", "items": "Paneer bhurji or tofu with 2 rotis, bowl of dal, warm turmeric milk" if is_veg else "Grilled fish fillet / chicken with 2 rotis, dal, warm turmeric milk"}
                ]
                option_b_meals = [
                    {"time": "07:30 AM", "name": "Budget Muscle Breakfast", "items": "Rolled oats cooked in milk with mashed banana and roasted seeds" if is_veg else "4 boiled eggs (2 whole, 2 whites) with 2 rotis and tea"},
                    {"time": "11:00 AM", "name": "Mid-Morning Snack", "items": "1 glass fresh buttermilk (chaas) with roasted cumin, 1 apple or seasonal fruit"},
                    {"time": "01:30 PM", "name": "Budget High-Protein Lunch", "items": "Soya chunk pulao (60g soya chunks) with onion-cucumber raita and dal"},
                    {"time": "04:30 PM", "name": "Pre-Workout Fuel", "items": "1 banana with black coffee"},
                    {"time": "06:45 PM", "name": "Post-Workout Low-Cost Fuel", "items": "40g Chana sattu shake or 3 boiled egg whites"},
                    {"time": "08:45 PM", "name": "Overnight Dinner", "items": "3 rotis with paneer or thick dal, green vegetables, warm milk"}
                ]
                student_sched = "Day: Academic Focus | 05:30 PM - 06:45 PM: Core & Strength Gym Session"
                athlete_sched = "10:00 AM - 11:30 AM: Kinetic Chain Gym | Afternoon Thoracic Mobility"

            else:
                day_type = "active_recovery"
                badge = "Rest & Recovery"
                title = "Spinal Decompression & Active Regeneration"
                ground_session = None
                gym_session = {
                    "status": "Active Recovery & Mobility",
                    "duration": "30 mins",
                    "focus": "Spinal elongation, thoracic rotation & soft tissue release",
                    "exercises": [
                        {"name": "Dead Hangs from Pullup Bar", "sets": 3, "reps": "45s"},
                        {"name": "Foam Rolling Hamstrings & IT Band", "sets": 3, "reps": "60s each"},
                        {"name": "Thoracic Spine Foam Roller Extensions", "sets": 3, "reps": "10 breaths"}
                    ],
                }
                nutrition_cals = max(2200, target_cals - 400)
                day_protein = base_protein
                day_carbs = round((nutrition_cals * 0.40) / 4)
                day_fats = round((nutrition_cals * 0.32) / 9)
                workload_adaptation = (
                    "Anti-inflammatory, lower glycemic profile with high antioxidant density to flush lactic acid "
                    "and rebuild collagen in knee tendons and lumbar cartilage."
                )
                option_a_meals = [
                    {"time": "08:00 AM", "name": "Antioxidant Morning Bowl", "items": "Chia seed pudding with almond milk, blueberries, flaxseeds, walnuts, and honey"},
                    {"time": "01:00 PM", "name": "Digestive Reset Lunch", "items": "Yellow moong dal khichdi with roasted paneer, steamed broccoli, curd, cucumber salad" if is_veg else "Clear chicken bone broth soup with steamed brown rice, veggies, salad"},
                    {"time": "04:30 PM", "name": "Mineral Hydration Booster", "items": "Tender coconut water, handful of roasted pumpkin seeds, green tea"},
                    {"time": "07:30 PM", "name": "Cellular Repair Dinner", "items": "Tofu or paneer stir-fry with zucchini and bell peppers, 2 rotis, chamomile tea" if is_veg else "Steamed fish with lemon herb butter, asparagus, 2 rotis, chamomile tea"}
                ]
                option_b_meals = [
                    {"time": "08:00 AM", "name": "Budget Clean Morning", "items": "Warm water with lemon and honey, 1 bowl daliya or oats porridge with milk"},
                    {"time": "01:00 PM", "name": "Soothing Recovery Lunch", "items": "Moong dal khichdi with curd, roasted papad, and tomato salad"},
                    {"time": "04:30 PM", "name": "Hydration Refresh", "items": "Nimbu pani (lemonade) with pinch of rock salt and roasted chana"},
                    {"time": "07:30 PM", "name": "Restorative Budget Dinner", "items": "2 soft chapatis with lauki/tori vegetable sabzi, light dal, warm milk with turmeric"}
                ]
                student_sched = "Full rest from high-impact bowling. Study & 20 min evening gentle walk."
                athlete_sched = "Ice baths, contrast therapy, massage & pool mobility."

            active_sched = student_sched if athlete_status == "student" else athlete_sched

            schedule.append({
                "day": day_name,
                "day_type": day_type,
                "badge": badge,
                "title": title,
                "workload": {
                    "ground_session": ground_session,
                    "gym_session": gym_session,
                },
                "nutrition": {
                    "calories": nutrition_cals,
                    "protein_g": day_protein,
                    "carbs_g": day_carbs,
                    "fats_g": day_fats,
                    "focus": "High glycogen replenishment" if is_match_spell else "Standard athletic balance",
                    "workload_adaptation": workload_adaptation,
                    "meals": option_a_meals,
                    "meal_options": {
                        "option_a": {
                            "name": "Option A: Elite High-Performance Fuel",
                            "description": "High-leucine, high-glycogen timing calibrated for maximum bowling velocity and rapid tissue repair.",
                            "meals": option_a_meals
                        },
                        "option_b": {
                            "name": "Option B: Budget-Friendly & Quick-Prep Fuel",
                            "description": "Affordable, easily available Indian ingredients (Sattu, Moong dal, Chana, Eggs/Soya) providing identical macro totals.",
                            "meals": option_b_meals
                        }
                    }
                },
                "student_schedule": student_sched,
                "fulltime_schedule": athlete_sched,
                "active_schedule": active_sched,
                "session_gap_advisory": (
                    f"{session_gap}. Maintain hydration and electrolytes."
                    if (has_ground and has_gym)
                    else "Single session focus today."
                ),
            })

        return schedule

    # -----------------------------------------------------------------
    # 4. SINGLE-CALL LLM GENERATED PERSONALIZED PLAN
    # -----------------------------------------------------------------

    def generate_plan(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes a single LLM call to generate a personalized 7-day Workout Plan,
        7-day Diet Plan, optional Supplements, Budget analysis, and Safety Notes.
        """
        is_valid, error_msg = self.validate_user_data(user_data)
        if not is_valid:
            return {
                "status": "error",
                "error_type": "ValidationError",
                "message": f"Validation failed: {error_msg}",
            }

        try:
            bmr = self.calculate_bmr(
                weight_kg=user_data["weight_kg"],
                height_cm=user_data["height_cm"],
                age=user_data["age"],
                sex=user_data["sex"],
            )

            tdee = self.calculate_tdee(
                bmr=bmr,
                activity_level=user_data["activity_level"],
            )

            target_calories = self.calculate_target_calories(
                tdee=tdee,
                goal=user_data["goal"],
            )

            structured_input = self.build_llm_input(
                user_data=user_data,
                bmr=bmr,
                tdee=tdee,
                target_calories=target_calories,
            )

            raw_response = self.call_llm_api(structured_input)
            parsed_plan = self._clean_and_parse_json(raw_response)

            if isinstance(parsed_plan, dict):
                parsed_plan["status"] = "success"
                return parsed_plan
            else:
                return {
                    "status": "error",
                    "error_type": "JSONParseError",
                    "message": "LLM output is not a valid JSON object.",
                }

        except json.JSONDecodeError as e:
            return {
                "status": "error",
                "error_type": "JSONParseError",
                "message": f"Failed to parse LLM JSON response: {str(e)}",
            }
        except ValueError as e:
            return {
                "status": "error",
                "error_type": "ValueError",
                "message": str(e),
            }
        except Exception as e:
            return {
                "status": "error",
                "error_type": "GenerationError",
                "message": f"An unexpected error occurred while generating the plan: {str(e)}",
            }

    # -----------------------------------------------------------------
    # STATIC & HELPER METHODS
    # -----------------------------------------------------------------

    @staticmethod
    def calculate_bmr(weight_kg: float, height_cm: float, age: int, sex: str) -> float:
        """
        Basal Metabolic Rate via Mifflin-St Jeor equation.
        """
        sex_clean = sex.lower().strip()
        if sex_clean == "male":
            bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + 5
        else:
            bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) - 161
        return round(bmr, 2)

    @staticmethod
    def calculate_tdee(bmr: float, activity_level: str) -> float:
        """
        Total Daily Energy Expenditure = BMR * activity_multiplier
        """
        level_clean = activity_level.lower().strip()
        multiplier = ACTIVITY_MULTIPLIERS.get(level_clean, 1.2)
        return round(bmr * multiplier, 2)

    @staticmethod
    def calculate_target_calories(tdee: float, goal: str) -> float:
        """
        Adjusts TDEE according to athletic goal.
        """
        goal_clean = goal.lower().strip()
        if "fat_loss" in goal_clean or "weight_loss" in goal_clean or "cut" in goal_clean:
            target = tdee * 0.85
        elif "muscle" in goal_clean or "hypertrophy" in goal_clean or "bulk" in goal_clean:
            target = tdee * 1.12
        elif "maintenance" in goal_clean or "maintain" in goal_clean:
            target = tdee
        elif "speed" in goal_clean or "performance" in goal_clean or "stamina" in goal_clean:
            target = tdee * 1.05
        else:
            target = tdee
        return round(target)

    @staticmethod
    def validate_user_data(user_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validates user profile dictionary fields.
        """
        if not isinstance(user_data, dict):
            return False, "User data must be a dictionary."

        age = user_data.get("age")
        if age is None or not isinstance(age, (int, float)) or age < 10 or age > 120:
            return False, "Invalid age: Age must be a number between 10 and 120."

        sex = user_data.get("sex")
        if not sex or not isinstance(sex, str) or sex.lower() not in VALID_SEXES:
            return False, f"Invalid sex: Sex must be one of {VALID_SEXES}."

        height_cm = user_data.get("height_cm")
        if height_cm is None or not isinstance(height_cm, (int, float)) or height_cm < 50 or height_cm > 270:
            return False, "Invalid height: Height must be a number between 50 cm and 270 cm."

        weight_kg = user_data.get("weight_kg")
        if weight_kg is None or not isinstance(weight_kg, (int, float)) or weight_kg < 20 or weight_kg > 300:
            return False, "Invalid weight: Weight must be a number between 20 kg and 300 kg."

        goal = user_data.get("goal")
        if not goal or not isinstance(goal, str) or len(goal.strip()) == 0:
            return False, "Invalid goal: Goal must be a non-empty string."

        activity_level = user_data.get("activity_level")
        if not activity_level or not isinstance(activity_level, str) or activity_level.lower() not in VALID_ACTIVITY_LEVELS:
            return False, f"Invalid activity level: Must be one of {VALID_ACTIVITY_LEVELS}."

        food_preference = user_data.get("food_preference")
        if not food_preference or not isinstance(food_preference, str):
            return False, "Invalid food preference: Must be a string (e.g., 'vegetarian' or 'non-veg')."

        budget = user_data.get("budget")
        if not budget or not isinstance(budget, dict):
            return False, "Invalid budget: Budget must be a dictionary containing amount, currency, and period."

        amount = budget.get("amount")
        if amount is None or not isinstance(amount, (int, float)) or amount <= 0:
            return False, "Invalid budget amount: Must be a positive number."

        currency = budget.get("currency")
        if not currency or not isinstance(currency, str):
            return False, "Invalid budget currency: Must be a non-empty string (e.g., 'INR', 'USD')."

        period = budget.get("period")
        if not period or not isinstance(period, str):
            return False, "Invalid budget period: Must be a non-empty string (e.g., 'month')."

        allergies = user_data.get("allergies", [])
        if not isinstance(allergies, list):
            return False, "Invalid allergies format: Allergies must be a list of strings."

        injuries = user_data.get("injuries", [])
        if not isinstance(injuries, list):
            return False, "Invalid injuries format: Injuries must be a list of injury objects."

        if "athlete_status" in user_data:
            status = str(user_data["athlete_status"]).lower().strip()
            if status not in ["student", "full_time_athlete", "athlete", "working_professional"]:
                return False, "Invalid athlete_status: Must be 'student' or 'full_time_athlete'."

        if "gym_days_per_week" in user_data:
            gym_days = user_data["gym_days_per_week"]
            if not isinstance(gym_days, (int, float)) or gym_days < 0 or gym_days > 7:
                return False, "Invalid gym_days_per_week: Must be an integer between 0 and 7."

        if "ground_days_per_week" in user_data:
            ground_days = user_data["ground_days_per_week"]
            if not isinstance(ground_days, (int, float)) or ground_days < 0 or ground_days > 7:
                return False, "Invalid ground_days_per_week: Must be an integer between 0 and 7."

        if "max_overs_days" in user_data:
            max_overs = user_data["max_overs_days"]
            if not isinstance(max_overs, (int, float)) or max_overs < 0 or max_overs > 7:
                return False, "Invalid max_overs_days: Must be an integer between 0 and 7."

        return True, None

    @staticmethod
    def build_llm_input(
        user_data: Dict[str, Any], bmr: float, tdee: float, target_calories: float
    ) -> Dict[str, Any]:
        """
        Constructs the structured JSON payload for LLM prompt generation.
        """
        return {
            "user_profile": {
                "age": user_data["age"],
                "sex": user_data["sex"].lower(),
                "height_cm": user_data["height_cm"],
                "weight_kg": user_data["weight_kg"],
                "goal": user_data["goal"],
                "activity_level": user_data["activity_level"].lower(),
            },
            "workload_schedule": {
                "athlete_status": user_data.get("athlete_status", "student"),
                "gym_days_per_week": user_data.get("gym_days_per_week", 3),
                "ground_days_per_week": user_data.get("ground_days_per_week", 4),
                "max_overs_days": user_data.get("max_overs_days", 2),
                "session_gap": user_data.get("session_gap", "Morning ground, Evening gym (4+ hours gap)"),
            },
            "nutrition_calculation": {
                "bmr": bmr,
                "tdee": tdee,
                "target_calories": target_calories,
            },
            "food_preferences": {
                "type": user_data.get("food_preference", "non-veg"),
                "allergies": user_data.get("allergies", []),
            },
            "budget": user_data.get("budget", {}),
            "injuries": user_data.get("injuries", []),
        }

    @staticmethod
    def call_llm_api(structured_input: Dict[str, Any]) -> str:
        """
        Calls the LLM API using the unified PacePro AI configuration
        (OpenAI, Anthropic, or Gemini depending on AI_PROVIDER).
        """
        if not config.is_llm_configured():
            raise ValueError(
                "API Key missing! Please configure 'AI_API_KEY' in your .env file."
            )

        workload = structured_input.get("workload_schedule", {})
        athlete_status = workload.get("athlete_status", "student")
        gym_days = workload.get("gym_days_per_week", 3)
        ground_days = workload.get("ground_days_per_week", 4)
        max_overs_days = workload.get("max_overs_days", 2)
        session_gap = workload.get("session_gap", "4+ hours gap")

        system_prompt = (
            "You are an expert AI Fast Bowling Coach, Sports Nutritionist, and Biomechanics Advisor for PacePro AI.\n"
            "Analyze the athlete profile, workload schedule, and nutritional calculations, then generate a synchronized "
            "7-day Workout Plan, Workload Schedule, and 7-day Diet Plan.\n"
            "Respond ONLY with a valid JSON object matching the standard PacePro plan schema."
        )

        prompt = f"""Analyze the athlete profile, workload schedule, and nutritional calculations, then generate a synchronized 7-day Workout Plan, Workload Schedule, and 7-day Diet Plan.

INPUT DATA:
{json.dumps(structured_input, indent=2)}

REQUIREMENTS & CONSTRAINTS:
1. ATHLETE ADAPTABILITY ({athlete_status.upper()}):
   - Design realistic schedule for '{athlete_status}'.
2. WORKLOAD & CRICKET BOWLING SPECIFICS:
   - Ground Days: Exactly {ground_days} days.
   - Gym Days: Exactly {gym_days} days.
   - Max Overs Bowling Days: Exactly {max_overs_days} days.
   - Session Gap: '{session_gap}'.
3. DIET & NUTRITION SYNCHRONIZATION:
   - Synchronize calories & carbs with bowling spell days.
   - FOOD PREFERENCE: '{structured_input.get('food_preferences', {}).get('type', 'standard')}'.
   - ALLERGIES: {json.dumps(structured_input.get('food_preferences', {}).get('allergies', []))}.
   - BUDGET: {structured_input.get('budget', {}).get('amount')} {structured_input.get('budget', {}).get('currency', 'INR')}.
4. SAFETY & LUMBAR PROTECTION:
   - Check injuries: {json.dumps(structured_input.get('injuries', []))}.
5. OPTIONAL SUPPLEMENTS:
   - Within budget and cricket performance focused.
6. OUTPUT FORMAT:
   - Raw JSON object only, matching standard PacePro plan schema.
"""
        client = LLMClient(max_tokens=4000)
        return client.generate_text(system_prompt=system_prompt, user_prompt=prompt)

    # -----------------------------------------------------------------
    # INTERNAL HELPERS
    # -----------------------------------------------------------------

    @staticmethod
    def _clean_and_parse_json(text: str) -> Dict[str, Any]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        return json.loads(cleaned)

    @staticmethod
    def _validate_dietary_inputs(age, weight_kg, height_cm, sex, activity_level, goal):
        if not (5 <= age <= 120):
            raise ValueError("age must be between 5 and 120.")
        if not (20 <= weight_kg <= 300):
            raise ValueError("weight_kg looks out of a plausible range.")
        if not (50 <= height_cm <= 270):
            raise ValueError("height_cm looks out of a plausible range.")
        if sex not in ("male", "female"):
            raise ValueError("sex must be 'male' or 'female'.")
        act_clean = activity_level.lower().strip()
        if act_clean not in ACTIVITY_MULTIPLIERS and act_clean not in config.ACTIVITY_MULTIPLIERS:
            raise ValueError(f"activity_level must be one of {list(ACTIVITY_MULTIPLIERS.keys())}")

    @staticmethod
    def _parse_and_score_sessions(sessions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        parsed = []
        for s in sessions:
            intensity = s.get("intensity")
            if intensity not in config.INTENSITY_RPE_FACTOR:
                raise ValueError(
                    f"intensity must be one of {list(config.INTENSITY_RPE_FACTOR)}, got {intensity!r}"
                )

            duration = s.get("duration_minutes")
            if duration is None or duration < 0:
                raise ValueError("duration_minutes must be a non-negative number.")

            load = duration * config.INTENSITY_RPE_FACTOR[intensity]
            parsed.append({
                "date": datetime.strptime(s["date"], "%Y-%m-%d").date(),
                "load": load,
            })
        return parsed

    @staticmethod
    def _average_daily_load(parsed_sessions, ref_date, window_days):
        window_start = ref_date - timedelta(days=window_days - 1)
        loads_in_window = [
            s["load"] for s in parsed_sessions
            if window_start <= s["date"] <= ref_date
        ]
        return sum(loads_in_window) / window_days

    @staticmethod
    def _determine_trend(current_week_load, previous_week_load):
        if previous_week_load == 0:
            return "insufficient_history" if current_week_load == 0 else "increasing"
        change_ratio = current_week_load / previous_week_load
        if change_ratio > 1.1:
            return "increasing"
        if change_ratio < 0.9:
            return "decreasing"
        return "stable"

    @staticmethod
    def _interpret_acwr(acwr):
        if acwr is None:
            return "unknown", "Not enough training history yet to calculate a ratio."

        if acwr > config.ACWR_SPIKE:
            return (
                "spike",
                f"Recent load (ACWR={acwr}) is well above the typical range "
                f"({config.ACWR_LOW}-{config.ACWR_HIGH}). Consider easing "
                f"intensity/volume and monitoring recovery closely.",
            )

        if acwr > config.ACWR_HIGH:
            return (
                "high",
                f"Recent load (ACWR={acwr}) is above the typical range. "
                f"Keep an eye on fatigue and recovery over the next few sessions.",
            )

        if acwr < config.ACWR_LOW:
            return (
                "low",
                f"Recent load (ACWR={acwr}) is below the typical range, "
                f"which can mean undertraining/detraining risk if sustained.",
            )

        return (
            "typical",
            f"Recent load (ACWR={acwr}) is within the commonly-cited "
            f"typical range ({config.ACWR_LOW}-{config.ACWR_HIGH}).",
        )


# =====================================================================
# SINGLETON INSTANCE & STANDALONE MODULE EXPORTS
# (Ensures full backward compatibility with existing code & unit tests)
# =====================================================================

planner_service = PlannerService()

validate_user_data = PlannerService.validate_user_data
calculate_bmr = PlannerService.calculate_bmr
calculate_tdee = PlannerService.calculate_tdee
calculate_target_calories = PlannerService.calculate_target_calories
build_llm_input = PlannerService.build_llm_input
call_llm_api = PlannerService.call_llm_api
generate_plan = planner_service.generate_plan
generate_combined_plan = planner_service.generate_combined_plan
calculate_workload = planner_service.calculate_workload
calculate_targets = planner_service.calculate_targets
suggest_meals = planner_service.suggest_meals
generate_diet_plan = planner_service.generate_diet_plan


# =====================================================================
# DEMO CLI
# =====================================================================

if __name__ == "__main__":
    print("--- Running PacePro AI Unified Planner Service ---\n")
    sample_user_data = {
        "age": 20,
        "sex": "male",
        "height_cm": 170,
        "weight_kg": 65,
        "goal": "increase_speed",
        "activity_level": "moderate",
        "food_preference": "non-veg",
        "allergies": ["peanuts"],
        "budget": {
            "amount": 5000,
            "currency": "INR",
            "period": "month",
        },
        "injuries": [
            {
                "body_part": "left_knee",
                "status": "pain",
                "severity": "mild",
            }
        ],
    }

    print("1. Testing Validation & Calculations...")
    valid, msg = validate_user_data(sample_user_data)
    print(f"   Validation Result: {valid} ({msg})")

    bmr_val = calculate_bmr(65, 170, 20, "male")
    tdee_val = calculate_tdee(bmr_val, "moderate")
    target_val = calculate_target_calories(tdee_val, "increase_speed")

    print(f"   BMR: {bmr_val} kcal")
    print(f"   TDEE: {tdee_val} kcal")
    print(f"   Target Calories: {target_val} kcal\n")

    print("2. Testing Workload ACWR...")
    sample_sessions = [
        {"date": "2026-09-01", "duration_minutes": 60, "intensity": "high"},
        {"date": "2026-09-03", "duration_minutes": 45, "intensity": "max"},
        {"date": "2026-09-05", "duration_minutes": 90, "intensity": "medium"},
    ]
    wl = planner_service.calculate_workload(sample_sessions, reference_date="2026-09-06")
    print(f"   ACWR: {wl.get('acwr')} | Zone: {wl.get('zone')} | Trend: {wl.get('trend')}\n")

    print("3. Testing Combined Plan Generation...")
    comb = planner_service.generate_combined_plan({"weight": 68, "height": 175, "athlete_status": "student"})
    print(f"   Combined Plan ID: {comb.get('id')} | Days: {len(comb.get('schedule', []))}\n")
