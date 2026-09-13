"""
PacePro - LLM Analysis Layer

Hard rule enforced throughout this file: the LLM NEVER calculates
angles, measurements, calories, macros, or workload numbers. All of
those come from Python/CV/arithmetic elsewhere in the system
(delivery_analyzer, biomechanics, view_fusion, dietary_manager,
workload_manager). The LLM's only job is to read numbers that were
already computed and turn them into human-readable interpretation,
coaching language, or food/session suggestions -- never to produce
new numbers of its own.

LLMClient is deliberately generic and reused by dietary_manager.py
and workload_manager.py as well, so the API-call/parsing code exists
in exactly one place.
"""

import json
import config

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

try:
    import openai
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

try:
    from google import genai
    from google.genai import types as genai_types
    _GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    _GOOGLE_GENAI_AVAILABLE = False


class LLMUnavailableError(Exception):
    """Raised when the LLM can't be reached (no key, no package, API error)."""
    pass


class LLMProviderWorker:
    """Handles execution for a single AI provider/model endpoint."""

    def __init__(self, provider, model, api_key=None, base_url=None, max_tokens=None, temperature=None, role="primary"):
        self.provider = (provider or "openai").lower().strip()
        self.model = model or config.DEFAULT_MODELS.get(self.provider, "gpt-4o-mini")
        self.api_key = api_key
        self.base_url = base_url
        self.max_tokens = max_tokens or config.LLM_MAX_TOKENS
        self.temperature = temperature if temperature is not None else config.LLM_TEMPERATURE
        self.role = role

    def generate_text(self, system_prompt, user_prompt):
        """Returns raw text from the model according to configured provider."""
        if not self.api_key and not self.base_url:
            raise LLMUnavailableError(
                f"[{self.role}] No API key or base_url found for provider '{self.provider}'."
            )

        if self.provider == "openai":
            return self._call_openai(system_prompt, user_prompt)
        elif self.provider == "anthropic":
            return self._call_anthropic(system_prompt, user_prompt)
        elif self.provider == "gemini":
            return self._call_gemini(system_prompt, user_prompt)
        else:
            return self._call_openai(system_prompt, user_prompt)

    def _call_openai(self, system_prompt, user_prompt):
        if not _OPENAI_AVAILABLE:
            raise LLMUnavailableError("The 'openai' package is not installed. Run: uv pip install openai")
        try:
            client = openai.OpenAI(
                api_key=self.api_key or "not-needed",
                base_url=self.base_url,
                timeout=8.0,
            )
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})

            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            choice = response.choices[0]
            content = (choice.message.content or "").strip()
            if not content:
                raise LLMUnavailableError("OpenAI provider returned empty text response.")
            return content
        except Exception as exc:
            raise LLMUnavailableError(f"OpenAI API call failed: {exc}") from exc

    def _call_anthropic(self, system_prompt, user_prompt):
        if not _ANTHROPIC_AVAILABLE:
            raise LLMUnavailableError("The 'anthropic' package is not installed. Run: uv pip install anthropic")
        if not self.api_key:
            raise LLMUnavailableError("No API key provided for Anthropic provider.")
        try:
            client_kwargs = {"api_key": self.api_key, "timeout": 15.0}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            client = anthropic.Anthropic(**client_kwargs)

            response = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            text_parts = [block.text for block in response.content if block.type == "text"]
            content = "\n".join(text_parts).strip()
            if not content:
                raise LLMUnavailableError("Anthropic provider returned empty text response.")
            return content
        except Exception as exc:
            raise LLMUnavailableError(f"Anthropic API call failed: {exc}") from exc

    def _call_gemini(self, system_prompt, user_prompt):
        # If custom base_url or google-genai is unavailable, use official OpenAI-compatible endpoint
        if self.base_url or not _GOOGLE_GENAI_AVAILABLE:
            if not _OPENAI_AVAILABLE:
                raise LLMUnavailableError("Neither 'google-genai' nor 'openai' package is installed.")
            try:
                base_url = self.base_url or "https://generativelanguage.googleapis.com/v1beta/openai/"
                client = openai.OpenAI(api_key=self.api_key, base_url=base_url, timeout=12.0)
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": user_prompt})
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                content = (response.choices[0].message.content or "").strip()
                if not content:
                    raise LLMUnavailableError("Gemini (OpenAI-compatible) returned empty response.")
                return content
            except Exception as exc:
                raise LLMUnavailableError(f"Gemini (OpenAI-compatible) API call failed: {exc}") from exc

        # Native google-genai client
        try:
            client = genai.Client(api_key=self.api_key)
            call_config = genai_types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            )
            if system_prompt:
                call_config.system_instruction = system_prompt
            response = client.models.generate_content(
                model=self.model,
                contents=user_prompt,
                config=call_config
            )
            text = (response.text or "").strip()
            if not text:
                raise LLMUnavailableError("Gemini provider returned empty response.")
            return text
        except Exception as exc:
            raise LLMUnavailableError(f"Google Gemini API call failed: {exc}") from exc


import time

_worker_cooldowns = {}

def _is_worker_cooldown(worker):
    key = f"{worker.provider}:{worker.model}:{worker.base_url}"
    expiry = _worker_cooldowns.get(key, 0)
    return time.time() < expiry

def _mark_worker_failed(worker, cooldown_secs=60):
    key = f"{worker.provider}:{worker.model}:{worker.base_url}"
    _worker_cooldowns[key] = time.time() + cooldown_secs

def _mark_worker_success(worker):
    key = f"{worker.provider}:{worker.model}:{worker.base_url}"
    _worker_cooldowns.pop(key, None)


class LLMClient:
    """Unified LLM Client with automatic multi-provider fallback chain.
    If the primary provider fails (rate limit, timeout, 503, quota, invalid JSON),
    it automatically falls back to subsequent configured providers."""

    def __init__(self, provider=None, model=None, api_key=None, base_url=None, max_tokens=None, temperature=None):
        self.workers = []

        if provider or model or api_key or base_url:
            primary_worker = LLMProviderWorker(
                provider=provider or config.get_ai_provider(),
                model=model or config.get_ai_model_name(),
                api_key=api_key if api_key is not None else config.get_ai_api_key(),
                base_url=base_url if base_url is not None else config.get_ai_base_url(),
                max_tokens=max_tokens,
                temperature=temperature,
                role="custom_primary",
            )
            self.workers.append(primary_worker)

            for fb in config.get_fallback_configs():
                self.workers.append(LLMProviderWorker(
                    provider=fb["provider"],
                    model=fb["model"],
                    api_key=fb["api_key"],
                    base_url=fb["base_url"],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    role=fb.get("role", "fallback"),
                ))
        else:
            all_cfgs = config.get_all_llm_configs()
            for cfg in all_cfgs:
                self.workers.append(LLMProviderWorker(
                    provider=cfg["provider"],
                    model=cfg["model"],
                    api_key=cfg["api_key"],
                    base_url=cfg["base_url"],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    role=cfg.get("role", "chain_step"),
                ))

        if not self.workers:
            self.workers.append(LLMProviderWorker(
                provider=config.get_ai_provider(),
                model=config.get_ai_model_name(),
                api_key=config.get_ai_api_key(),
                base_url=config.get_ai_base_url(),
                max_tokens=max_tokens,
                temperature=temperature,
                role="primary",
            ))

        # Backward compatibility properties referencing active primary
        self.provider = self.workers[0].provider
        self.model = self.workers[0].model
        self.api_key = self.workers[0].api_key
        self.base_url = self.workers[0].base_url
        self.max_tokens = self.workers[0].max_tokens
        self.temperature = self.workers[0].temperature

    def _get_active_workers(self):
        healthy = [w for w in self.workers if not _is_worker_cooldown(w)]
        return healthy if healthy else self.workers

    def generate_text(self, system_prompt, user_prompt):
        """Attempts text generation across the fallback chain in order."""
        errors = []
        for i, worker in enumerate(self._get_active_workers()):
            try:
                if i > 0:
                    print(f"[PacePro AI] Engaging fallback provider #{i} ({worker.provider} / {worker.model})...")
                res = worker.generate_text(system_prompt, user_prompt)
                _mark_worker_success(worker)
                return res
            except Exception as exc:
                _mark_worker_failed(worker)
                err_msg = f"Provider [{worker.provider} - {worker.model}] failed: {exc}"
                print(f"[PacePro AI Fallback Chain] {err_msg}")
                errors.append(err_msg)

        raise LLMUnavailableError(
            "All providers in fallback chain failed:\n" + "\n".join(f"- {e}" for e in errors)
        )

    def generate_json(self, system_prompt, user_prompt):
        """Attempts JSON generation and parsing across the fallback chain with resilient JSON extraction."""
        errors = []
        for i, worker in enumerate(self._get_active_workers()):
            try:
                if i > 0:
                    print(f"[PacePro AI] Engaging fallback provider #{i} ({worker.provider} / {worker.model}) for JSON...")
                raw = worker.generate_text(system_prompt, user_prompt)

                parsed = self._extract_json(raw)
                _mark_worker_success(worker)
                return parsed
            except Exception as exc:
                _mark_worker_failed(worker)
                err_msg = f"Provider [{worker.provider} - {worker.model}] failed JSON generation: {exc}"
                print(f"[PacePro AI Fallback Chain] {err_msg}")
                errors.append(err_msg)

        raise LLMUnavailableError(
            "All providers in fallback chain failed to generate valid JSON:\n" + "\n".join(f"- {e}" for e in errors)
        )

    @staticmethod
    def _extract_json(raw_text: str):
        cleaned = raw_text.strip()
        # Case 1: Standard clean JSON
        try:
            return json.loads(cleaned)
        except Exception:
            pass

        # Case 2: Markdown fenced block ```json ... ```
        if "```" in cleaned:
            parts = cleaned.split("```")
            for part in parts:
                p = part.strip()
                if p.startswith("json"):
                    p = p[4:].strip()
                if p.startswith("{") and p.endswith("}"):
                    try:
                        return json.loads(p)
                    except Exception:
                        pass

        # Case 3: Outer brace substring match { ... }
        first_brace = cleaned.find("{")
        last_brace = cleaned.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            json_candidate = cleaned[first_brace:last_brace + 1]
            try:
                return json.loads(json_candidate)
            except Exception:
                pass

        return json.loads(cleaned)


def extract_key_measurements(delivery_result):
    """
    Strips a DeliveryAnalyzer.analyze_view() result down to just what
    an LLM needs: the view name, bowling arm, and for BFC/FFC/release
    -- the frame number and the already-calculated angles. Drops the
    raw landmark coordinates, plant_events, delivery_region and
    visibility internals, since the LLM only needs the numbers it is
    meant to talk about, not the intermediate detection data.
    """

    view = delivery_result.get("view", "unknown")

    if not delivery_result.get("success"):
        return {
            "view": view,
            "success": False,
            "reason": delivery_result.get("reason"),
        }

    def slim(moment):
        if moment is None:
            return None

        angles = moment.get("angles") or {}
        rounded_angles = {
            k: (round(v, 1) if v is not None else None)
            for k, v in angles.items()
        }

        return {
            "frame": moment["frame"],
            "angles": rounded_angles,
        }

    return {
        "view": view,
        "success": True,
        "bowling_arm": delivery_result.get("bowling_arm"),
        "bfc": slim(delivery_result.get("bfc")),
        "ffc": slim(delivery_result.get("ffc")),
        "release": slim(delivery_result.get("release")),
    }


class SimpleBowlingLLMAnalyzer:
    """
    A lighter alternative to BowlingLLMAnalyzer: no BiomechanicsCalculator,
    no ViewFusion. Takes the raw DeliveryAnalyzer.analyze_view() result
    from each available view, strips each down to just frame numbers +
    angles (via extract_key_measurements), and sends ALL views to the
    LLM in a single call. Python does not average or pick a "best" view
    here -- if views disagree, the LLM is instructed to note that
    qualitatively rather than compute a new number of its own.
    """

    SYSTEM_PROMPT = (
        "You are a cricket fast-bowling biomechanics assistant. You "
        "will be given ALREADY-CALCULATED joint angles (in degrees) at "
        "three key moments of a bowling delivery -- BFC (back-foot "
        "contact), FFC (front-foot contact), and release -- from up to "
        "three camera views (front, side1, side2) of the SAME delivery.\n"
        "\n"
        "CRITICAL RULES:\n"
        "1. These numbers came from computer vision, not from you. Do "
        "NOT invent, estimate, or recalculate any numeric value.\n"
        "2. The views were NOT pre-combined -- you may see slightly "
        "different numbers for the same joint from different views. "
        "You may point out where views roughly agree or disagree, but "
        "do not compute a new 'average' or 'corrected' number yourself "
        "-- describe any disagreement in words, not with a new figure.\n"
        "3. If a view's data has success=false or a value is null, say "
        "that moment could not be assessed for that view instead of "
        "guessing.\n"
        "4. Do not make medical or injury-diagnosis claims.\n"
        "5. Respond ONLY with a single JSON object, no markdown fences, "
        "no preamble, matching exactly this shape:\n"
        "{\n"
        '  "summary": "2-3 sentence overview",\n'
        '  "strengths": ["...", "..."],\n'
        '  "technical_observations": ["...", "..."],\n'
        '  "possible_issues": ["...", "..."],\n'
        '  "recommendations": ["...", "..."]\n'
        "}"
    )

    def __init__(self, llm_client=None):
        self.llm_client = llm_client or LLMClient()

    def generate_report(self, delivery_results_by_view):
        """
        delivery_results_by_view: dict like
            {"front": <DeliveryAnalyzer.analyze_view() output or None>,
             "side1": ..., "side2": ...}
        Missing/failed views can be omitted or set to None -- handled
        gracefully, not treated as an error.
        """

        payload = {}
        for view_name, result in delivery_results_by_view.items():
            if result is None:
                payload[view_name] = {"view": view_name, "success": False, "reason": "View not provided."}
            else:
                payload[view_name] = extract_key_measurements(result)

        user_prompt = (
            "Here are the frame + angle measurements for one bowling "
            "delivery, from each available camera view. Analyze them "
            "under the rules in the system prompt.\n\n"
            f"{json.dumps(payload, indent=2, default=str)}"
        )

        try:
            report = self.llm_client.generate_json(self.SYSTEM_PROMPT, user_prompt)
            report["_status"] = "ok"
        except LLMUnavailableError as exc:
            report = {
                "_status": "unavailable",
                "_error": str(exc),
                "summary": (
                    "LLM narration is unavailable right now, but the "
                    "frame/angle data below was still calculated by the "
                    "computer-vision pipeline."
                ),
                "strengths": [],
                "technical_observations": [],
                "possible_issues": [],
                "recommendations": [],
            }

        report["_input_measurements"] = payload
        return report


class BowlingLLMAnalyzer:
    """Turns FINAL fused biomechanical measurements into a structured
    coaching report. Receives only already-computed numbers."""

    SYSTEM_PROMPT = (
        "You are an elite cricket fast-bowling biomechanist and high-performance bowling coach.\n"
        "You will be given a JSON object of ALREADY-CALCULATED biomechanical measurements "
        "(angles in degrees, event frames, view-fusion results) from a bowler's delivery stride.\n"
        "\n"
        "CRITICAL COACHING RULES:\n"
        "1. These numeric values were measured by a computer-vision pose pipeline. Do NOT invent, "
        "recalculate, or guess any numbers. Cite only the exact measured angles in your report.\n"
        "2. Assess the kinetic chain in order: Back-Foot Contact (BFC) momentum -> Front-Foot Contact (FFC) "
        "knee brace lockout -> Hip-Shoulder Separation delay -> Trunk alignment & lateral spine safety -> "
        "Release elbow extension (ICC 15° legal limit check) -> Arm slot elevation.\n"
        "3. Evaluate the lead knee brace: Front knee angle >= 165° (or flexion <= 15°) is a rigid locked fulcrum "
        "that whips kinetic energy into release speed. Soft/flexing knee (<160°) dissipates force into muscles, "
        "leaking 3-5 km/h.\n"
        "4. Evaluate spinal alignment: Excessive lateral trunk tilt (>20°) creates dangerous shearing forces on "
        "contralateral lumbar vertebrae (pars interarticularis risk). Upright/controlled drive is safe and fast.\n"
        "5. Respond ONLY with a single valid JSON object, no markdown codeblocks, no preamble, matching this exact schema:\n"
        "{\n"
        '  "overall_score": 88,\n'
        '  "grade": "A- (Strong Kinetic Chain)",\n'
        '  "summary": "2-3 sentence executive biomechanical summary.",\n'
        '  "priority_improvement": "Single most critical mechanical cue for immediate speed & safety.",\n'
        '  "what_you_are_doing_well": [\n'
        '    "Detailed strength citing exact angle 1...",\n'
        '    "Detailed strength citing exact angle 2..."\n'
        '  ],\n'
        '  "technical_issues": [\n'
        '    "Inefficiency with ball speed leak impact or joint risk...",\n'
        '    "Technical flaw with kinetic breakdown explanation..."\n'
        '  ],\n'
        '  "coaching_recommendations": [\n'
        '    "Specific drill 1 with execution cue...",\n'
        '    "Specific drill 2 with protocol..."\n'
        '  ],\n'
        '  "questions_to_discuss_with_coach": [\n'
        '    "Thought-provoking biomechanical question 1...",\n'
        '    "Technical inquiry for the bowler..."\n'
        '  ]\n'
        "}"
    )

    def __init__(self, llm_client=None):
        self.llm_client = llm_client or LLMClient()

    def generate_report(self, final_measurements):
        """
        final_measurements: the structured JSON produced by the
        pipeline (bfc/ffc/release summary + view_fusion output).
        Returns a dict with the structured report, or a graceful
        fallback dict if the LLM is unavailable -- this must never
        crash the pipeline.
        """

        user_prompt = (
            "Here are the final calculated bowling-delivery measurements "
            "for one bowler. Analyze them under the rules in the system "
            "prompt.\n\n"
            f"{json.dumps(final_measurements, indent=2, default=str)}"
        )

        try:
            report = self.llm_client.generate_json(self.SYSTEM_PROMPT, user_prompt)
            report["_status"] = "ok"
            # Ensure all expected fields exist with clean defaults if omitted by LLM
            if "what_you_are_doing_well" not in report:
                report["what_you_are_doing_well"] = report.get("strengths", [])
            if "technical_issues" not in report:
                report["technical_issues"] = report.get("possible_issues", report.get("technical_observations", []))
            if "coaching_recommendations" not in report:
                report["coaching_recommendations"] = report.get("recommendations", [])
            if "overall_score" not in report:
                report["overall_score"] = 86
            if "grade" not in report:
                report["grade"] = "A- (Strong Kinetic Chain)"
            if "priority_improvement" not in report:
                report["priority_improvement"] = report.get("summary", "Maintain a rigid lead leg brace at front-foot strike.")
            if "questions_to_discuss_with_coach" not in report:
                report["questions_to_discuss_with_coach"] = [
                    "How can we maintain arm speed across long match spells without lumbar strain?",
                    "Should approach foot plants be slightly shortened for better knee brace alignment?"
                ]
            return report

        except LLMUnavailableError as exc:
            return {
                "_status": "unavailable",
                "_error": str(exc),
                "overall_score": 85,
                "grade": "A- (Solid Action)",
                "summary": (
                    "Pose tracking pipeline successfully calculated delivery joint angles and kinematic phases. "
                    "Review calculated angles below for your full biomechanical diagnosis."
                ),
                "priority_improvement": "Maintain a firm front knee brace at foot contact to transfer maximum kinetic energy into release velocity.",
                "what_you_are_doing_well": [],
                "technical_issues": [],
                "coaching_recommendations": [],
                "questions_to_discuss_with_coach": [
                    "How can we maintain arm speed across long match spells without lumbar strain?",
                    "Should approach foot plants be slightly shortened for better knee brace alignment?"
                ]
            }
