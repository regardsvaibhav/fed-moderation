"""
GenAI Explainer — Gemini API
=============================
Takes a post + model prediction and returns:
  1. Plain-English explanation of WHY it was flagged
  2. Which specific words/phrases triggered it
  3. Which policy it violates
  4. Confidence reasoning

This is the layer that separates your project from every
other federated learning tutorial — explainability is the
#1 demand from enterprises deploying content moderation.

Usage:
    from src.genai.explainer import ModerationExplainer
    explainer = ModerationExplainer()
    result = explainer.explain("some post text", prediction=1, confidence=0.87)
"""

import os, sys, json, time
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from loguru import logger
from src.config import GEMINI_API_KEY, GEMINI_MODEL

# ── Try importing Gemini ────────────────────────────────────────────────────
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("google-generativeai not installed. Using fallback explainer.")


# ── Prompt Template ─────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert content moderation analyst for a social media platform.
Your job is to explain moderation decisions clearly and fairly.

When given a post and its moderation decision, you must respond ONLY with a JSON object
in exactly this format (no markdown, no extra text):
{
  "decision": "TOXIC" or "SAFE",
  "confidence_explanation": "one sentence explaining the confidence level",
  "flagged_phrases": ["phrase1", "phrase2"],
  "policy_violated": "name of policy or null if safe",
  "explanation": "2-3 sentence plain English explanation for why this decision was made",
  "target_group": "which group is targeted or null if safe",
  "severity": "HIGH", "MEDIUM", "LOW", or "NONE"
}"""

USER_PROMPT_TEMPLATE = """Post: "{text}"

Model Decision: {decision} (confidence: {confidence:.1%})
Model was trained using Federated Learning with Differential Privacy (ε={epsilon:.2f})

Analyze this post and provide the moderation explanation in the required JSON format."""


# ── Fallback Explainer (no API key needed) ──────────────────────────────────
def fallback_explain(text: str, prediction: int, confidence: float) -> dict:
    """
    Rule-based fallback when Gemini API is not available.
    Used during development / testing.
    """
    toxic_keywords = [
        'hate', 'kill', 'destroy', 'attack', 'stupid', 'idiot',
        'worthless', 'trash', 'filth', 'disgusting', 'ban', 'eliminate'
    ]
    words = text.lower().split()
    found_keywords = [w for w in words if any(k in w for k in toxic_keywords)]

    if prediction == 1:
        return {
            "decision": "TOXIC",
            "confidence_explanation": f"Model is {confidence:.1%} confident this content is harmful.",
            "flagged_phrases": found_keywords[:3] if found_keywords else [words[0] if words else "content"],
            "policy_violated": "Hate Speech & Harassment Policy",
            "explanation": (
                f"This post was flagged as potentially harmful content. "
                f"The federated model detected patterns associated with toxic language "
                f"with {confidence:.1%} confidence across privacy-preserving client nodes."
            ),
            "target_group": "Unspecified",
            "severity": "HIGH" if confidence > 0.85 else "MEDIUM",
            "source": "fallback"
        }
    else:
        return {
            "decision": "SAFE",
            "confidence_explanation": f"Model is {confidence:.1%} confident this content is safe.",
            "flagged_phrases": [],
            "policy_violated": None,
            "explanation": (
                f"This post was classified as safe content. "
                f"No policy violations were detected by the federated model "
                f"with {confidence:.1%} confidence."
            ),
            "target_group": None,
            "severity": "NONE",
            "source": "fallback"
        }


# ── Main Explainer Class ────────────────────────────────────────────────────
class ModerationExplainer:
    def __init__(self):
        self.use_gemini = False

        if GEMINI_AVAILABLE and GEMINI_API_KEY:
            try:
                genai.configure(api_key=GEMINI_API_KEY)
                self.model = genai.GenerativeModel(
                    model_name=GEMINI_MODEL
                )
                self.system_prompt = SYSTEM_PROMPT
                self.use_gemini = True
                logger.success("Gemini API connected ✅")
            except Exception as e:
                logger.warning(f"Gemini init failed: {e}. Using fallback.")
        else:
            logger.info("No Gemini API key found — using fallback explainer.")
            logger.info("Add GEMINI_API_KEY to .env for full GenAI explanations.")

    def explain(
        self,
        text: str,
        prediction: int,
        confidence: float,
        epsilon: float = 3.8,
        max_retries: int = 2,
    ) -> dict:
        """
        Generate explanation for a moderation decision.

        Args:
            text:       The post being moderated
            prediction: 0 (safe) or 1 (toxic)
            confidence: Model confidence score (0-1)
            epsilon:    Privacy budget consumed (for transparency)

        Returns:
            dict with decision, explanation, flagged_phrases, etc.
        """
        decision = "TOXIC" if prediction == 1 else "SAFE"

        # Use fallback if Gemini not available
        if not self.use_gemini:
            result = fallback_explain(text, prediction, confidence)
            result['epsilon'] = epsilon
            return result

        # Build prompt
        prompt = USER_PROMPT_TEMPLATE.format(
            text=text[:500],  # truncate very long posts
            decision=decision,
            confidence=confidence,
            epsilon=epsilon,
        )

        # Call Gemini with retry
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(self.system_prompt + "\n\n" + prompt)
                raw = response.text.strip()

                # Clean up response (remove markdown fences if present)
                if raw.startswith('```'):
                    raw = raw.split('```')[1]
                    if raw.startswith('json'):
                        raw = raw[4:]
                raw = raw.strip()

                result = json.loads(raw)
                result['epsilon'] = epsilon
                result['source'] = 'gemini'
                return result

            except json.JSONDecodeError:
                logger.warning(f"Gemini returned non-JSON (attempt {attempt+1}). Raw: {raw[:100]}")
            except Exception as e:
                logger.warning(f"Gemini API error (attempt {attempt+1}): {e}")
                time.sleep(1)

        # If all retries fail, use fallback
        logger.warning("All Gemini retries failed. Using fallback.")
        result = fallback_explain(text, prediction, confidence)
        result['epsilon'] = epsilon
        return result

    def explain_batch(self, posts: list[dict]) -> list[dict]:
        """
        Explain a batch of moderation decisions.
        posts: list of dicts with keys: text, prediction, confidence
        """
        results = []
        for post in posts:
            result = self.explain(
                text=post['text'],
                prediction=post['prediction'],
                confidence=post['confidence'],
                epsilon=post.get('epsilon', 3.8),
            )
            results.append(result)
            time.sleep(0.5)  # rate limiting
        return results


# ── Quick Test ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    explainer = ModerationExplainer()

    test_cases = [
        {
            "text": "I hate all people from that religion, they should be banned",
            "prediction": 1,
            "confidence": 0.92,
            "epsilon": 3.8
        },
        {
            "text": "Just finished an amazing book about history, highly recommend!",
            "prediction": 0,
            "confidence": 0.95,
            "epsilon": 3.8
        },
        {
            "text": "These people are destroying our country with their beliefs",
            "prediction": 1,
            "confidence": 0.74,
            "epsilon": 3.8
        }
    ]

    print("\n" + "="*60)
    print("MODERATION EXPLAINER TEST")
    print("="*60)

    for i, case in enumerate(test_cases):
        print(f"\n[Post {i+1}]: {case['text'][:60]}...")
        result = explainer.explain(**case)
        print(f"Decision   : {result['decision']}")
        print(f"Severity   : {result['severity']}")
        print(f"Policy     : {result['policy_violated']}")
        print(f"Flagged    : {result['flagged_phrases']}")
        print(f"Explanation: {result['explanation'][:120]}...")
        print(f"Source     : {result['source']}")
        print("-"*60)