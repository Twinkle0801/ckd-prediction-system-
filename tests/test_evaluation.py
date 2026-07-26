# tests/test_evaluation.py
import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from src.evaluation import (
    check_target_leakage, top_correlated_feature, compare_tuned_vs_untuned,
    flag_precision_collapse, select_final_model, explain_model,
    top_shap_features, explain_single_prediction, generate_model_card,
)


def test_check_target_leakage_excludes_id_column():
    df = pd.DataFrame({
        "id": range(10),
        "hemo": np.random.rand(10) * 15,
        "sc": np.random.rand(10) * 5,
        "classification": [1, 0] * 5,
    })
    corr = check_target_leakage(df)
    assert "id" not in corr.index
    assert "classification" not in corr.index
    assert set(corr.index) == {"hemo", "sc"}


def test_check_target_leakage_raises_on_non_numeric_target():
    df = pd.DataFrame({
        "hemo": np.random.rand(10) * 15,
        "classification": ["ckd", "notckd"] * 5,  # not encoded to 0/1
    })
    with pytest.raises(ValueError, match="not numeric"):
        check_target_leakage(df)


def test_check_target_leakage_sorts_by_absolute_correlation():
    n = 50
    strong = np.linspace(0, 1, n)
    weak = np.random.rand(n) * 0.01
    df = pd.DataFrame({
        "strong_feature": strong,
        "weak_feature": weak,
        "classification": strong.round().astype(int),  # near-perfect correlation
    })
    corr = check_target_leakage(df)
    assert corr.index[0] == "strong_feature"


def test_top_correlated_feature_returns_first_row():
    corr = pd.Series({"hemo": -0.9, "sc": 0.6})
    feature, value = top_correlated_feature(corr)
    assert feature == "hemo"
    assert value == -0.9


def test_compare_tuned_vs_untuned_filters_and_combines():
    baseline_df = pd.DataFrame({
        "model": ["logreg", "random_forest", "xgboost"],
        "stage": ["untuned"] * 3,
        "recall": [0.9, 0.95, 0.93],
    })
    tuned_df = pd.DataFrame({
        "model": ["random_forest", "xgboost"],
        "stage": ["tuned"] * 2,
        "recall": [0.97, 0.96],
        "best_params": [{"max_depth": 5}, {"max_depth": 3}],
    })
    compare_df = compare_tuned_vs_untuned(baseline_df, tuned_df, ["random_forest", "xgboost"])
    assert "logreg" not in compare_df["model"].values  # not a candidate, correctly excluded
    assert "best_params" not in compare_df.columns  # dropped
    assert len(compare_df) == 4  # 2 untuned + 2 tuned


def test_flag_precision_collapse_finds_low_precision_rows():
    compare_df = pd.DataFrame({"model": ["a", "b"], "precision": [0.9, 0.3]})
    flagged = flag_precision_collapse(compare_df, min_precision=0.5)
    assert list(flagged["model"]) == ["b"]


def test_select_final_model_unknown_name_raises():
    with pytest.raises(ValueError):
        select_final_model("not_a_model", {}, pd.DataFrame(), pd.Series())


def test_select_final_model_returns_scores_and_justification():
    X_test = pd.DataFrame(np.random.rand(20, 3), columns=list("abc"))
    y_test = pd.Series([0, 1] * 10)
    model = LogisticRegression(max_iter=1000).fit(X_test, y_test)
    tuned_models = {"logreg": model}
    final_model, final_scores, justification = select_final_model("logreg", tuned_models, X_test, y_test)
    assert final_model is model
    assert "recall" in final_scores
    assert "Final model: logreg" in justification
    assert f"Recall: {final_scores['recall']:.3f}" in justification


def test_explain_model_raises_for_unsupported_model_type():
    X_test = pd.DataFrame(np.random.rand(10, 3), columns=list("abc"))
    model = LogisticRegression(max_iter=1000).fit(X_test, pd.Series([0, 1] * 5))
    with pytest.raises(TypeError, match="TreeExplainer"):
        explain_model(model, X_test)


def test_explain_model_works_for_decision_tree():
    X_train = pd.DataFrame(np.random.rand(40, 3), columns=list("abc"))
    y_train = pd.Series([0, 1] * 20)
    model = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
    explainer, shap_values = explain_model(model, X_train)
    assert explainer is not None
    assert shap_values is not None


def test_top_shap_features_ranks_by_mean_abs_value():
    X_test = pd.DataFrame({"a": [0]*5, "b": [0]*5, "c": [0]*5})
    shap_values = np.array([
        [0.1, 5.0, 0.1],
        [0.1, -5.0, 0.1],
        [0.2, 4.5, -0.2],
        [-0.1, 5.5, 0.1],
        [0.1, -4.8, 0.05],
    ])  # 'b' clearly dominates
    top = top_shap_features(shap_values, X_test, n=2)
    assert top.index[0] == "b"


def test_explain_single_prediction_sorts_by_absolute_shap_value():
    X_test = pd.DataFrame({"a": [1.0], "b": [2.0], "c": [3.0]})
    model = DecisionTreeClassifier(random_state=42).fit(
        pd.DataFrame(np.random.rand(20, 3), columns=list("abc")), pd.Series([0, 1] * 10)
    )
    class FakeExplainer:
        expected_value = 0.4
    shap_values = np.array([[0.1, -0.9, 0.2]])
    result = explain_single_prediction(model, FakeExplainer(), shap_values, X_test, 0)
    assert result["top_contributions"][0]["feature"] == "b"  # largest |shap| value
    assert result["expected_value"] == 0.4


def test_generate_model_card_writes_file_and_includes_limitations(tmp_path):
    final_scores = {
        "accuracy": 0.97, "precision": 0.95, "recall": 0.98, "f1": 0.96,
        "roc_auc": 0.99, "confusion_matrix": [[10, 1], [0, 20]],
    }
    path = tmp_path / "MODEL_CARD.md"
    card_text = generate_model_card(
        final_model_name="xgboost", final_scores=final_scores,
        justification="Chosen for recall.", top_feature="hemo", top_corr_value=-0.85,
        n_train=100, n_test=25, n_features=20, path=str(path),
    )
    assert path.exists()
    assert "Not a diagnostic tool" in card_text
    assert "hemo" in card_text
    assert "0.980" in card_text  # recall formatted to 3 decimals