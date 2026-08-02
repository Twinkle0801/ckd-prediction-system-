# tests/test_ai_assistant.py
from src.ai_assistant.prompts import build_explainer_prompt, DISCLAIMER
from src.ai_assistant.grounding import check_grounding, extract_numbers
from src.ai_assistant.tools import call_shap_explainer_tool


def test_build_explainer_prompt_includes_all_top_factors():
    result = {"prediction_label": "ckd", "probability": 0.93}
    shap = {"top_contributions": [
        {"feature": "hemo", "value": 15.4, "shap_contribution": -0.42},
        {"feature": "sc", "value": 1.2, "shap_contribution": 0.31},
    ]}
    prompt = build_explainer_prompt(result, shap)
    assert "hemo" in prompt
    assert "sc" in prompt
    assert "15.4" in prompt
    assert "ONLY" in prompt  # the grounding constraint must be explicit


def test_extract_numbers_finds_decimals_and_integers():
    numbers = extract_numbers("The value was 15.4 and count was 3, negative -2.5 too.")
    assert "15.4" in numbers
    assert "3.0" in numbers
    assert "-2.5" in numbers


def test_check_grounding_catches_fabricated_numbers():
    shap = {"top_contributions": [{"feature": "hemo", "value": 15.4, "shap_contribution": -0.42}]}
    result = {"probability": 0.93}
    fabricated_text = "The hemoglobin of 12.0 suggests a 78 percent risk."
    grounding = check_grounding(fabricated_text, shap, result)
    assert grounding["grounded"] is False
    assert "12.0" in grounding["ungrounded_numbers"]
    assert "78.0" in grounding["ungrounded_numbers"]


def test_check_grounding_passes_when_only_real_numbers_used():
    shap = {"top_contributions": [{"feature": "hemo", "value": 15.4, "shap_contribution": -0.42}]}
    result = {"probability": 0.93}
    real_text = "The hemoglobin of 15.4 contributed -0.42 to the prediction, at a probability of 0.93."
    grounding = check_grounding(real_text, shap, result)
    assert grounding["grounded"] is True
    assert grounding["ungrounded_numbers"] == []


def test_call_shap_explainer_tool_always_includes_disclaimer():
    """Even in mock mode (no API key/credits), the disclaimer must be present --
    this is enforced in code, not dependent on the LLM's behavior."""
    result = {"prediction_label": "ckd", "probability": 0.93}
    shap = {"top_contributions": [{"feature": "hemo", "value": 15.4, "shap_contribution": -0.42}]}
    output = call_shap_explainer_tool(result, shap)
    assert DISCLAIMER in output["explanation"]


def test_call_shap_explainer_tool_returns_grounded_flag():
    result = {"prediction_label": "ckd", "probability": 0.93}
    shap = {"top_contributions": [{"feature": "hemo", "value": 15.4, "shap_contribution": -0.42}]}
    output = call_shap_explainer_tool(result, shap)
    assert "grounded" in output
    assert isinstance(output["grounded"], bool)