"""
Story 13: Translate Model Findings Into Specific Interventions

Tasks covered:
#67 Recommendation 1: Driver behaviour interventions.
#68 Recommendation 2: Lighting / road-environment interventions.
#69 Recommendation 3: Enforcement interventions.
#70 Recommendation 4: Vulnerable road user interventions.
#71 Recommendation 5: Temporal / seasonal deployment interventions.
#72 Recommendation 6: Predictive deployment recommendation.

This script consumes outputs from:
- Story 4: logistic regression coefficients and odds ratios
- Story 8: XGBoost metrics and SHAP feature importance
- Story 10: geospatial ward and intersection hotspots

It does NOT train a new model.
It translates prior model findings into practical intervention recommendations.
"""

import argparse
from pathlib import Path

import pandas as pd


# =========================================================
# Helpers
# =========================================================

def ensure_output_dir(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def read_csv_if_exists(path):
    path = Path(path)

    if path.exists():
        return pd.read_csv(path)

    print(f"Warning: missing optional file: {path}")
    return pd.DataFrame()


def load_story4_outputs(story4_dir):
    story4_dir = Path(story4_dir)

    significant_predictors = read_csv_if_exists(
        story4_dir / "task_25_significant_predictors.csv"
    )

    coefficients = read_csv_if_exists(
        story4_dir / "task_25_logistic_regression_coefficients.csv"
    )

    return significant_predictors, coefficients


def load_story8_outputs(story8_dir):
    story8_dir = Path(story8_dir)

    shap_importance = read_csv_if_exists(
        story8_dir / "task_48_shap_feature_importance.csv"
    )

    xgb_metrics = read_csv_if_exists(
        story8_dir / "task_47_xgboost_metrics.csv"
    )

    model_comparison = read_csv_if_exists(
        story8_dir / "task_47_model_comparison_with_xgboost.csv"
    )

    return shap_importance, xgb_metrics, model_comparison


def load_story10_outputs(story10_dir):
    story10_dir = Path(story10_dir)

    ward_stats = read_csv_if_exists(
        story10_dir / "task_56_ward_fatality_stats.csv"
    )

    top_intersections = read_csv_if_exists(
        story10_dir / "task_57_top10_fatal_intersections.csv"
    )

    return ward_stats, top_intersections


def get_or_value(coefficients, keyword):
    """
    Find odds ratio from Story 4 coefficient table using a keyword.
    """
    if coefficients.empty or "Predictor" not in coefficients.columns:
        return None

    matched = coefficients[
        coefficients["Predictor"].astype(str).str.lower().str.contains(keyword.lower(), na=False)
    ]

    if matched.empty or "OR" not in matched.columns:
        return None

    return matched.iloc[0]["OR"]


def get_shap_rank(shap_importance, keyword):
    """
    Find SHAP rank from Story 8 feature importance table using a keyword.
    """
    if shap_importance.empty or "feature" not in shap_importance.columns:
        return None

    shap_importance = shap_importance.reset_index(drop=True).copy()
    shap_importance["rank"] = shap_importance.index + 1

    matched = shap_importance[
        shap_importance["feature"].astype(str).str.lower().str.contains(keyword.lower(), na=False)
    ]

    if matched.empty:
        return None

    return int(matched.iloc[0]["rank"])


def top_list(df, column, n=5):
    if df.empty or column not in df.columns:
        return []

    return df[column].dropna().astype(str).head(n).tolist()


def safe_round(value, decimals=2):
    try:
        return round(float(value), decimals)
    except Exception:
        return "N/A"


# =========================================================
# Recommendation builders
# =========================================================

def build_recommendation_67(coefficients, shap_importance):
    """
    #67 Driver behaviour recommendation.
    """
    aggressive_or = get_or_value(coefficients, "aggressive")
    distracted_or = get_or_value(coefficients, "distracted")
    drivcond_rank = get_shap_rank(shap_importance, "drivcond")

    recommendation = {
        "task": 67,
        "recommendation_area": "Driver behaviour",
        "evidence_source": "Story 4 logistic regression + Story 8 SHAP",
        "model_finding": (
            "Behaviour-related variables such as aggressive driving, distraction, "
            "and driver condition appeared in statistical or model-interpretation outputs."
        ),
        "specific_intervention": (
            "Target high-risk driver behaviours through focused public education, "
            "impaired-driving checks, distracted-driving enforcement, and aggressive-driving deterrence."
        ),
        "implementation_detail": (
            "Prioritize campaigns and enforcement near high-risk corridors and during time windows "
            "identified by model and temporal patterns. Messages should focus on impairment, distraction, "
            "speeding/aggression, and failure to yield."
        ),
        "evidence_detail": (
            f"Aggressive OR={safe_round(aggressive_or)}; "
            f"Distracted OR={safe_round(distracted_or)}; "
            f"Driver-condition SHAP rank={drivcond_rank if drivcond_rank else 'N/A'}."
        ),
        "expected_impact": (
            "Reduce fatal collision risk linked to preventable driver behaviours."
        ),
    }

    return pd.DataFrame([recommendation])


def build_recommendation_68(coefficients, shap_importance):
    """
    #68 Lighting recommendation.
    """
    daylight_or = get_or_value(coefficients, "light_Daylight")
    light_rank = get_shap_rank(shap_importance, "light")

    recommendation = {
        "task": 68,
        "recommendation_area": "Lighting and road environment",
        "evidence_source": "Story 4 logistic regression + Story 8 SHAP",
        "model_finding": (
            "Lighting variables contributed to fatal/non-fatal prediction differences. "
            "Daylight was associated with lower fatal collision odds compared with the dark reference category."
        ),
        "specific_intervention": (
            "Upgrade street lighting at high-risk corridors, intersections, pedestrian crossings, "
            "bus stops, and approaches to major arterial roads."
        ),
        "implementation_detail": (
            "Prioritize locations where fatal collisions cluster under poor lighting or where vulnerable "
            "road users are exposed. Use brighter lighting, improved crosswalk visibility, reflective signage, "
            "and signal visibility improvements."
        ),
        "evidence_detail": (
            f"Daylight OR={safe_round(daylight_or)}; "
            f"lighting SHAP rank={light_rank if light_rank else 'N/A'}."
        ),
        "expected_impact": (
            "Improve visibility and reduce collision severity during dark or low-visibility conditions."
        ),
    }

    return pd.DataFrame([recommendation])


def build_recommendation_69(coefficients, shap_importance, top_intersections):
    """
    #69 Enforcement recommendation.
    """
    red_light_or = get_or_value(coefficients, "red_light")
    high_risk_intersections = top_list(top_intersections, "Intersection", n=5)

    recommendation = {
        "task": 69,
        "recommendation_area": "Targeted enforcement",
        "evidence_source": "Story 4 logistic regression + Story 10 intersection hotspots",
        "model_finding": (
            "Fatal collisions are concentrated at specific intersections and corridors. "
            "Behavioural and traffic-control factors can be translated into targeted enforcement."
        ),
        "specific_intervention": (
            "Deploy targeted enforcement and automated safety monitoring at high-risk intersections."
        ),
        "implementation_detail": (
            "Focus on red-light running, speeding, failure to yield, impaired driving, and unsafe turning "
            "movements. Priority locations include: "
            + "; ".join(high_risk_intersections)
            if high_risk_intersections
            else "Focus on the top fatal-intersection list from Story 10."
        ),
        "evidence_detail": (
            f"Red-light OR={safe_round(red_light_or)}; "
            f"Top intersections used from Story 10 hotspot outputs."
        ),
        "expected_impact": (
            "Reduce severe collision risk where enforcement can address repeated high-risk behaviour."
        ),
    }

    return pd.DataFrame([recommendation])


def build_recommendation_70(coefficients, shap_importance):
    """
    #70 Vulnerable road user recommendation.
    """
    pedestrian_or = get_or_value(coefficients, "pedestrian")
    cyclist_or = get_or_value(coefficients, "cyclist")
    motorcyclist_or = get_or_value(coefficients, "motorcyclist")

    pedestrian_rank = get_shap_rank(shap_importance, "pedestrian")
    cyclist_rank = get_shap_rank(shap_importance, "cyclist")
    motorcycle_rank = get_shap_rank(shap_importance, "motorcyclist")

    recommendation = {
        "task": 70,
        "recommendation_area": "Vulnerable road users",
        "evidence_source": "Story 4 logistic regression + Story 8 SHAP",
        "model_finding": (
            "Pedestrian, cyclist, and motorcyclist indicators are relevant for identifying vulnerable "
            "road-user risk in KSI collisions."
        ),
        "specific_intervention": (
            "Prioritize pedestrian and cyclist protection at high-risk corridors and intersections."
        ),
        "implementation_detail": (
            "Use protected bike lanes, leading pedestrian intervals, raised crosswalks, curb extensions, "
            "turn-calming treatments, pedestrian refuge islands, and protected signal phases."
        ),
        "evidence_detail": (
            f"Pedestrian OR={safe_round(pedestrian_or)}, SHAP rank={pedestrian_rank if pedestrian_rank else 'N/A'}; "
            f"Cyclist OR={safe_round(cyclist_or)}, SHAP rank={cyclist_rank if cyclist_rank else 'N/A'}; "
            f"Motorcyclist OR={safe_round(motorcyclist_or)}, SHAP rank={motorcycle_rank if motorcycle_rank else 'N/A'}."
        ),
        "expected_impact": (
            "Reduce exposure and injury severity for pedestrians, cyclists, and motorcyclists."
        ),
    }

    return pd.DataFrame([recommendation])


def build_recommendation_71(coefficients, shap_importance):
    """
    #71 Temporal and seasonal recommendation.
    """
    year_rank = get_shap_rank(shap_importance, "year")
    month_rank = get_shap_rank(shap_importance, "month")
    hour_rank = get_shap_rank(shap_importance, "hour")

    recommendation = {
        "task": 71,
        "recommendation_area": "Temporal and seasonal deployment",
        "evidence_source": "Story 8 SHAP + engineered temporal features from Story 1",
        "model_finding": (
            "Temporal variables such as year, month, and hour can influence the model's fatality predictions."
        ),
        "specific_intervention": (
            "Use time-sensitive deployment of safety resources during high-risk periods."
        ),
        "implementation_detail": (
            "Schedule enforcement, signal timing review, patrol visibility, and public safety campaigns "
            "around time windows where fatal collision risk is elevated. Reassess timing plans seasonally."
        ),
        "evidence_detail": (
            f"Year SHAP rank={year_rank if year_rank else 'N/A'}; "
            f"Month SHAP rank={month_rank if month_rank else 'N/A'}; "
            f"Hour SHAP rank={hour_rank if hour_rank else 'N/A'}."
        ),
        "expected_impact": (
            "Improve timing of interventions instead of applying the same resources evenly across all periods."
        ),
    }

    return pd.DataFrame([recommendation])


def build_recommendation_72(xgb_metrics, shap_importance, ward_stats):
    """
    #72 Predictive deployment recommendation.
    """
    auc = "N/A"
    recall = "N/A"

    if not xgb_metrics.empty:
        if "roc_auc" in xgb_metrics.columns:
            auc = safe_round(xgb_metrics["roc_auc"].iloc[0], 3)
        if "recall" in xgb_metrics.columns:
            recall = safe_round(xgb_metrics["recall"].iloc[0], 3)

    high_risk_wards = top_list(ward_stats, "wardname", n=5)

    top_features = []
    if not shap_importance.empty and "feature" in shap_importance.columns:
        top_features = shap_importance["feature"].head(5).astype(str).tolist()

    recommendation = {
        "task": 72,
        "recommendation_area": "Predictive deployment",
        "evidence_source": "Story 8 XGBoost performance + Story 10 hotspot analysis",
        "model_finding": (
            "The XGBoost model can help identify higher-risk records and features, while geospatial analysis "
            "identifies where interventions should be prioritized."
        ),
        "specific_intervention": (
            "Deploy a predictive monitoring workflow for Toronto Transportation Services to prioritize "
            "inspections, enforcement planning, signal review, and corridor safety audits."
        ),
        "implementation_detail": (
            "Use model outputs as a decision-support layer, not an automatic decision maker. Combine predicted "
            "risk with ward/intersection hotspots, recent collision trends, and engineering judgment. "
            "Priority wards include: "
            + "; ".join(high_risk_wards)
            if high_risk_wards
            else "Use Story 10 ward and intersection hotspot outputs for location prioritization."
        ),
        "evidence_detail": (
            f"XGBoost ROC-AUC={auc}; Fatal-class recall={recall}; "
            f"Top SHAP features={', '.join(top_features) if top_features else 'N/A'}."
        ),
        "expected_impact": (
            "Support proactive resource allocation before severe collisions recur at known high-risk locations."
        ),
    }

    return pd.DataFrame([recommendation])


# =========================================================
# Reporting
# =========================================================

def save_recommendation_outputs(
    output_dir,
    rec67,
    rec68,
    rec69,
    rec70,
    rec71,
    rec72,
):
    rec67.to_csv(output_dir / "task_67_driver_behaviour_recommendation.csv", index=False)
    rec68.to_csv(output_dir / "task_68_lighting_recommendation.csv", index=False)
    rec69.to_csv(output_dir / "task_69_enforcement_recommendation.csv", index=False)
    rec70.to_csv(output_dir / "task_70_vulnerable_road_user_recommendation.csv", index=False)
    rec71.to_csv(output_dir / "task_71_temporal_seasonal_recommendation.csv", index=False)
    rec72.to_csv(output_dir / "task_72_predictive_deployment_recommendation.csv", index=False)

    all_recommendations = pd.concat(
        [rec67, rec68, rec69, rec70, rec71, rec72],
        ignore_index=True,
    )

    all_recommendations.to_csv(
        output_dir / "task_67_72_all_recommendations.csv",
        index=False,
    )

    return all_recommendations


def write_markdown_report(all_recommendations, output_dir):
    lines = []

    lines.append("# Story 13: Translate Model Findings Into Specific Interventions")
    lines.append("")
    lines.append(
        "This report translates model findings from logistic regression, XGBoost SHAP, "
        "and geospatial hotspot analysis into specific road-safety interventions."
    )
    lines.append("")
    lines.append("## Evidence Sources")
    lines.append("")
    lines.append("- Story 4: Logistic regression odds ratios and significant predictors")
    lines.append("- Story 8: XGBoost performance and SHAP feature importance")
    lines.append("- Story 10: Ward-level fatality rates and top fatal intersections")
    lines.append("")
    lines.append("## Recommendations")
    lines.append("")

    for _, row in all_recommendations.iterrows():
        lines.append(f"### Task #{row['task']}: {row['recommendation_area']}")
        lines.append("")
        lines.append(f"**Model finding:** {row['model_finding']}")
        lines.append("")
        lines.append(f"**Recommended intervention:** {row['specific_intervention']}")
        lines.append("")
        lines.append(f"**Implementation detail:** {row['implementation_detail']}")
        lines.append("")
        lines.append(f"**Evidence detail:** {row['evidence_detail']}")
        lines.append("")
        lines.append(f"**Expected impact:** {row['expected_impact']}")
        lines.append("")

    lines.append("## Caution")
    lines.append("")
    lines.append(
        "These recommendations are based on statistical associations, machine-learning explanations, "
        "and hotspot analysis. They should support, not replace, transportation engineering judgment, "
        "community consultation, and field safety audits."
    )
    lines.append("")

    report_text = "\n".join(lines)

    report_path = output_dir / "task_67_72_recommendations_report.md"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    return report_path


# =========================================================
# Main
# =========================================================

def main(story4_dir, story8_dir, story10_dir, output_dir):
    output_dir = ensure_output_dir(output_dir)

    print("Loading Story 4 outputs...")
    significant_predictors, coefficients = load_story4_outputs(story4_dir)

    print("Loading Story 8 outputs...")
    shap_importance, xgb_metrics, model_comparison = load_story8_outputs(story8_dir)

    print("Loading Story 10 outputs...")
    ward_stats, top_intersections = load_story10_outputs(story10_dir)

    print("\nBuilding Story 13 recommendations...")

    rec67 = build_recommendation_67(coefficients, shap_importance)
    rec68 = build_recommendation_68(coefficients, shap_importance)
    rec69 = build_recommendation_69(coefficients, shap_importance, top_intersections)
    rec70 = build_recommendation_70(coefficients, shap_importance)
    rec71 = build_recommendation_71(coefficients, shap_importance)
    rec72 = build_recommendation_72(xgb_metrics, shap_importance, ward_stats)

    all_recommendations = save_recommendation_outputs(
        output_dir,
        rec67,
        rec68,
        rec69,
        rec70,
        rec71,
        rec72,
    )

    report_path = write_markdown_report(all_recommendations, output_dir)

    print("\nStory 13 complete.")
    print(f"Outputs saved to: {output_dir}")
    print("\nGenerated files:")
    print("- task_67_driver_behaviour_recommendation.csv")
    print("- task_68_lighting_recommendation.csv")
    print("- task_69_enforcement_recommendation.csv")
    print("- task_70_vulnerable_road_user_recommendation.csv")
    print("- task_71_temporal_seasonal_recommendation.csv")
    print("- task_72_predictive_deployment_recommendation.csv")
    print("- task_67_72_all_recommendations.csv")
    print(f"- {report_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Story 13: Translate model findings into specific interventions."
    )

    parser.add_argument(
        "--story4-dir",
        required=True,
        help="Directory containing Story 4 outputs."
    )

    parser.add_argument(
        "--story8-dir",
        required=True,
        help="Directory containing Story 8 outputs."
    )

    parser.add_argument(
        "--story10-dir",
        required=True,
        help="Directory containing Story 10 outputs."
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where Story 13 outputs will be saved."
    )

    args = parser.parse_args()

    main(
        story4_dir=args.story4_dir,
        story8_dir=args.story8_dir,
        story10_dir=args.story10_dir,
        output_dir=args.output_dir,
    )