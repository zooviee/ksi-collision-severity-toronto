"""
KSI Collision Severity Predictor — Streamlit App
Factors Affecting Traffic Collision Severity in Toronto
Group 5 | DAMO-699-5 | University of Niagara Falls Canada

Usage:
    # Auto-discovers task_54_best_model.pkl in outputs/ or any outputs/story-N/ subfolder
    streamlit run src/app.py

    # Or point explicitly to a specific model file
    streamlit run src/app.py -- --model outputs/story-9/task_54_best_model.pkl
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="KSI Collision Severity Predictor",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

CORE_FEATURES = [
    "invage", "older_adult", "school_child", "motorcyclist",
    "aggressive", "distracted", "red_light", "hour", "is_weekend",
    "light_Dark", "light_Dark with Artificial Lighting", "light_Dusk",
    "rdsfcond_Wet", "rdsfcond_Ice", "rdsfcond_Loose Snow",
    "traffictl_Traffic Signal", "traffictl_Stop Sign",
    "road_class_Expressway", "road_class_Local", "road_class_Minor Arterial",
    "accloc_Non-Intersection", "accloc_Intersection-Related",
    "impactype_Cyclist Collision", "impactype_Rear End",
    "impactype_Turning Movement",
]

FEAT_LABELS = {
    "invage":                              "Age of person involved",
    "older_adult":                         "Older adult (65+)",
    "school_child":                        "School-age child",
    "motorcyclist":                        "Motorcyclist involved",
    "aggressive":                          "Aggressive driving",
    "distracted":                          "Distracted driving",
    "red_light":                           "Red-light violation",
    "hour":                                "Hour of day (0–23)",
    "is_weekend":                          "Weekend collision",
    "light_Dark":                          "Dark — no artificial light",
    "light_Dark with Artificial Lighting": "Dark — artificial lighting present",
    "light_Dusk":                          "Dusk conditions",
    "rdsfcond_Wet":                        "Wet road surface",
    "rdsfcond_Ice":                        "Icy road surface",
    "rdsfcond_Loose Snow":                 "Loose snow on road",
    "traffictl_Traffic Signal":            "Traffic signal (vs. No control)",
    "traffictl_Stop Sign":                 "Stop sign (vs. No control)",
    "road_class_Expressway":               "Expressway road",
    "road_class_Local":                    "Local road",
    "road_class_Minor Arterial":           "Minor arterial road",
    "accloc_Non-Intersection":             "Non-intersection location",
    "accloc_Intersection-Related":         "Intersection-related location",
    "impactype_Cyclist Collision":         "Cyclist collision type",
    "impactype_Rear End":                  "Rear-end collision",
    "impactype_Turning Movement":          "Turning movement collision",
}

# Project root — always resolved relative to this file, not the working directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _find_model() -> Path:
    """
    Search for task_54_best_model.pkl in order:
      1. outputs/task_54_best_model.pkl          (flat layout)
      2. outputs/story-9/task_54_best_model.pkl  (story-subfolder layout)
      3. Any outputs/story-*/task_54_best_model.pkl (any story subfolder)
    Returns the first match found.
    """
    candidates = [
        PROJECT_ROOT / "outputs" / "task_54_best_model.pkl",
        PROJECT_ROOT / "outputs" / "story-9" / "task_54_best_model.pkl",
    ]
    # Also scan any story-N subfolders dynamically
    outputs_dir = PROJECT_ROOT / "outputs"
    if outputs_dir.exists():
        for sub in sorted(outputs_dir.iterdir()):
            p = sub / "task_54_best_model.pkl"
            if p not in candidates:
                candidates.append(p)

    for p in candidates:
        if p.exists():
            return p
    return candidates[0]   # return primary path so error message is informative


DEFAULT_MODEL_PATH = _find_model()


# ─────────────────────────────────────────────────────────────────────────────
# Load model (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)


# ─────────────────────────────────────────────────────────────────────────────
# Feature builder from widget values
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_vector(
    age: int,
    hour: int,
    is_weekend: bool,
    light_cond: str,
    road_surface: str,
    road_class: str,
    traffic_control: str,
    collision_location: str,
    impact_type: str,
    older_adult: bool,
    school_child: bool,
    motorcyclist: bool,
    aggressive: bool,
    distracted: bool,
    red_light: bool,
) -> pd.DataFrame:

    row = {f: 0 for f in CORE_FEATURES}

    # Numeric
    row["invage"]     = float(age)
    row["hour"]       = float(hour)
    row["is_weekend"] = int(is_weekend)

    # Boolean flags
    row["older_adult"]  = int(older_adult)
    row["school_child"] = int(school_child)
    row["motorcyclist"] = int(motorcyclist)
    row["aggressive"]   = int(aggressive)
    row["distracted"]   = int(distracted)
    row["red_light"]    = int(red_light)

    # Lighting (reference = Daylight)
    light_map = {
        "Daylight":                         None,
        "Dark (no artificial lighting)":    "light_Dark",
        "Dark with artificial lighting":    "light_Dark with Artificial Lighting",
        "Dusk":                             "light_Dusk",
    }
    lf = light_map.get(light_cond)
    if lf:
        row[lf] = 1

    # Road surface (reference = Dry)
    surface_map = {
        "Dry":         None,
        "Wet":         "rdsfcond_Wet",
        "Ice":         "rdsfcond_Ice",
        "Loose Snow":  "rdsfcond_Loose Snow",
    }
    sf = surface_map.get(road_surface)
    if sf:
        row[sf] = 1

    # Road class (reference = Major Arterial)
    rc_map = {
        "Major Arterial": None,
        "Expressway":     "road_class_Expressway",
        "Local":          "road_class_Local",
        "Minor Arterial": "road_class_Minor Arterial",
    }
    rf = rc_map.get(road_class)
    if rf:
        row[rf] = 1

    # Traffic control (reference = No Control)
    tc_map = {
        "No Control":      None,
        "Traffic Signal":  "traffictl_Traffic Signal",
        "Stop Sign":       "traffictl_Stop Sign",
    }
    tf = tc_map.get(traffic_control)
    if tf:
        row[tf] = 1

    # Collision location (reference = At Intersection)
    loc_map = {
        "At Intersection":       None,
        "Non-Intersection":      "accloc_Non-Intersection",
        "Intersection-Related":  "accloc_Intersection-Related",
    }
    lof = loc_map.get(collision_location)
    if lof:
        row[lof] = 1

    # Impact type (reference = Angle)
    imp_map = {
        "Angle":             None,
        "Cyclist Collision": "impactype_Cyclist Collision",
        "Rear End":          "impactype_Rear End",
        "Turning Movement":  "impactype_Turning Movement",
    }
    imf = imp_map.get(impact_type)
    if imf:
        row[imf] = 1

    return pd.DataFrame([row])[CORE_FEATURES]


# ─────────────────────────────────────────────────────────────────────────────
# Risk gauge figure
# ─────────────────────────────────────────────────────────────────────────────

def make_gauge(prob: float) -> plt.Figure:
    """Semicircular gauge coloured by risk level."""
    fig, ax = plt.subplots(figsize=(4, 2.5),
                           subplot_kw={"aspect": "equal"},
                           facecolor="#0E1117")
    ax.set_facecolor("#0E1117")

    # Draw arc segments: green → amber → red
    theta  = np.linspace(np.pi, 0, 300)
    r_out, r_in = 1.0, 0.65

    def arc(t1, t2, color, alpha=1.0):
        seg   = np.linspace(t1, t2, 100)
        x_out = np.cos(seg) * r_out
        y_out = np.sin(seg) * r_out
        x_in  = np.cos(seg[::-1]) * r_in
        y_in  = np.sin(seg[::-1]) * r_in
        ax.fill(np.concatenate([x_out, x_in]),
                np.concatenate([y_out, y_in]),
                color=color, alpha=alpha, zorder=2)

    # Background track
    arc(np.pi, 0, "#2C3E50", alpha=0.6)

    # Coloured fill up to probability
    needle_theta = np.pi * (1 - prob)
    if prob <= 0.33:
        arc(np.pi, needle_theta, "#27AE60")
    elif prob <= 0.60:
        arc(np.pi, min(needle_theta, np.pi * 0.67), "#27AE60")
        if needle_theta < np.pi * 0.67:
            arc(np.pi * 0.67, needle_theta, "#E67E22")
        else:
            arc(np.pi * 0.67, needle_theta, "#E67E22")
    else:
        arc(np.pi, np.pi * 0.67, "#27AE60")
        arc(np.pi * 0.67, np.pi * 0.40, "#E67E22")
        arc(np.pi * 0.40, needle_theta, "#C0392B")

    # Needle
    nx = np.cos(needle_theta) * 0.82
    ny = np.sin(needle_theta) * 0.82
    ax.annotate("", xy=(nx, ny), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="white",
                                lw=2.5, mutation_scale=16))
    ax.plot(0, 0, "o", color="white", markersize=8, zorder=5)

    # Labels
    pct   = prob * 100
    color = "#27AE60" if prob < 0.33 else "#E67E22" if prob < 0.60 else "#C0392B"
    label = "LOW RISK" if prob < 0.33 else "MODERATE RISK" if prob < 0.60 else "HIGH RISK"

    ax.text(0, -0.15, f"{pct:.1f}%", ha="center", va="center",
            fontsize=22, fontweight="bold", color=color)
    ax.text(0, -0.38, f"FATAL RISK — {label}", ha="center", va="center",
            fontsize=8, color=color, fontweight="bold")

    # Tick labels
    for t, txt in [(np.pi, "0%"), (np.pi*0.5, "50%"), (0, "100%")]:
        ax.text(np.cos(t)*1.12, np.sin(t)*1.12, txt,
                ha="center", va="center", fontsize=7, color="#95A5A6")

    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-0.6, 1.15)
    ax.axis("off")
    fig.tight_layout(pad=0.1)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SHAP bar figure
# ─────────────────────────────────────────────────────────────────────────────

def make_shap_bar(shap_vals: np.ndarray,
                  feature_values: pd.DataFrame) -> plt.Figure:
    sv   = pd.Series(shap_vals, index=CORE_FEATURES)
    top3 = sv.abs().nlargest(3).index.tolist()

    labels = [FEAT_LABELS.get(f, f) for f in top3]
    values = [sv[f] for f in top3]
    fvals  = [feature_values[f].iloc[0] for f in top3]

    colors = ["#C0392B" if v > 0 else "#2980B9" for v in values]

    fig, ax = plt.subplots(figsize=(5, 2.4), facecolor="#0E1117")
    ax.set_facecolor("#0E1117")

    bars = ax.barh(range(3), values, color=colors,
                   edgecolor="#2C3E50", height=0.55, zorder=3)

    ax.set_yticks(range(3))
    ax.set_yticklabels(
        [f"{l}\n(value={fvals[i]:.0f})" for i, l in enumerate(labels)],
        fontsize=8, color="white"
    )
    ax.axvline(0, color="#95A5A6", linewidth=1.2, zorder=4)

    for bar, v in zip(bars, values):
        ax.text(v + (0.01 if v >= 0 else -0.01),
                bar.get_y() + bar.get_height() / 2,
                f"{v:+.3f}",
                va="center",
                ha="left" if v >= 0 else "right",
                fontsize=8, color="white", fontweight="bold")

    ax.set_xlabel("SHAP value  (+ = increases fatal risk)", fontsize=8,
                  color="#95A5A6")
    ax.set_title("Top 3 SHAP Contributors", fontsize=9,
                 fontweight="bold", color="white", pad=6)
    ax.tick_params(colors="#95A5A6", labelsize=7.5)
    ax.invert_yaxis()

    for spine in ax.spines.values():
        spine.set_edgecolor("#2C3E50")
    ax.grid(axis="x", color="#2C3E50", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout(pad=0.5)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:linear-gradient(90deg,#1A1A2E,#16213E);
                padding:20px 28px; border-radius:10px; margin-bottom:16px;
                border-left:5px solid #C0392B;">
        <h1 style="color:white; margin:0; font-size:1.7rem;">
            🚦 KSI Collision Severity Predictor
        </h1>
        <p style="color:#95A5A6; margin:4px 0 0 0; font-size:0.9rem;">
            Toronto Motor Vehicle Collisions (2006–2026) &nbsp;|&nbsp;
            XGBoost model (AUC = 0.8624) &nbsp;|&nbsp;
            Group 5 · DAMO-699-5 · University of Niagara Falls Canada
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load model ────────────────────────────────────────────────────────────
    model_path = DEFAULT_MODEL_PATH
    if not model_path.exists():
        st.error(
            f"❌ **Model file not found.**\n\n"
            f"Searched in `{PROJECT_ROOT / 'outputs'}` and all story subfolders.\n\n"
            f"**Fix options:**\n"
            f"- Run Story 9: `python src/model_selection_cv.py --output-dir outputs/story-9 ...`\n"
            f"- Or pass the model path directly: "
            f"`streamlit run src/app.py -- --model path/to/task_54_best_model.pkl`"
        )
        st.stop()

    st.sidebar.caption(f"Model: `{model_path.relative_to(PROJECT_ROOT)}`")
    model     = load_model(str(model_path))
    explainer = get_explainer(model)

    # ── Sidebar — input widgets ───────────────────────────────────────────────
    st.sidebar.header("🔧 Collision Scenario")
    st.sidebar.markdown("---")

    st.sidebar.subheader("👤 Person Involved")
    age         = st.sidebar.slider("Age of person involved", 0, 100, 35, 1)
    older_adult = st.sidebar.checkbox("Older adult (65+)",
                                       value=(age >= 65))
    school_child = st.sidebar.checkbox("School-age child", value=False)
    motorcyclist = st.sidebar.checkbox("Motorcyclist", value=False)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🕐 Time")
    hour       = st.sidebar.slider("Hour of day (0 = midnight, 12 = noon)", 0, 23, 14)
    is_weekend = st.sidebar.checkbox("Weekend (Sat/Sun)", value=False)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🌦 Environment")
    light_cond  = st.sidebar.selectbox(
        "Lighting condition",
        ["Daylight", "Dark (no artificial lighting)",
         "Dark with artificial lighting", "Dusk"],
    )
    road_surface = st.sidebar.selectbox(
        "Road surface condition",
        ["Dry", "Wet", "Ice", "Loose Snow"],
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🛣 Infrastructure")
    road_class = st.sidebar.selectbox(
        "Road classification",
        ["Major Arterial", "Expressway", "Local", "Minor Arterial"],
    )
    traffic_control = st.sidebar.selectbox(
        "Traffic control at location",
        ["No Control", "Traffic Signal", "Stop Sign"],
    )
    collision_location = st.sidebar.selectbox(
        "Collision location",
        ["At Intersection", "Non-Intersection", "Intersection-Related"],
    )
    impact_type = st.sidebar.selectbox(
        "Impact type",
        ["Angle", "Cyclist Collision", "Rear End", "Turning Movement"],
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚠️ Behaviour")
    aggressive = st.sidebar.checkbox("Aggressive driving flagged", value=False)
    distracted = st.sidebar.checkbox("Distracted driving flagged", value=False)
    red_light  = st.sidebar.checkbox("Red-light violation", value=False)

    st.sidebar.markdown("---")
    predict_btn = st.sidebar.button("🔍 Predict Fatal Risk", type="primary",
                                     use_container_width=True)

    # ── Main content area ─────────────────────────────────────────────────────
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.subheader("Scenario Summary")

        # Show selected scenario as a readable table
        scenario_data = {
            "Parameter":  ["Age", "Hour", "Weekend", "Lighting", "Road Surface",
                           "Road Class", "Traffic Control", "Location",
                           "Impact Type", "Older Adult", "Motorcyclist",
                           "Aggressive", "Distracted", "Red Light"],
            "Value": [str(age), f"{hour}:00",
                      "Yes" if is_weekend else "No",
                      light_cond, road_surface, road_class,
                      traffic_control, collision_location, impact_type,
                      "Yes" if older_adult else "No",
                      "Yes" if motorcyclist else "No",
                      "Yes" if aggressive else "No",
                      "Yes" if distracted else "No",
                      "Yes" if red_light else "No"],
        }
        st.dataframe(pd.DataFrame(scenario_data), hide_index=True,
                     use_container_width=True,
                     height=min(450, 35 * len(scenario_data["Parameter"]) + 38))

    with col_right:
        if predict_btn or True:  # auto-predict on any change
            # Build feature vector
            X = build_feature_vector(
                age=age, hour=hour, is_weekend=is_weekend,
                light_cond=light_cond, road_surface=road_surface,
                road_class=road_class, traffic_control=traffic_control,
                collision_location=collision_location, impact_type=impact_type,
                older_adult=older_adult, school_child=school_child,
                motorcyclist=motorcyclist, aggressive=aggressive,
                distracted=distracted, red_light=red_light,
            )

            # Predict
            prob     = float(model.predict_proba(X.values)[0, 1])
            shap_vals = explainer.shap_values(X.values)[0]

            # ── Gauge ──────────────────────────────────────────────────────
            st.subheader("Fatal Risk Prediction")
            gauge_fig = make_gauge(prob)
            st.pyplot(gauge_fig, use_container_width=True)
            plt.close(gauge_fig)

            # Risk band alert
            if prob < 0.25:
                st.success(f"✅ **Low risk** — {prob*100:.1f}% predicted fatal probability. "
                           f"Conditions are relatively safe.")
            elif prob < 0.50:
                st.warning(f"⚠️ **Moderate risk** — {prob*100:.1f}% predicted fatal probability. "
                           f"One or more elevated risk factors present.")
            elif prob < 0.70:
                st.error(f"🔴 **High risk** — {prob*100:.1f}% predicted fatal probability. "
                         f"Multiple serious risk factors detected.")
            else:
                st.error(f"🚨 **Very high risk** — {prob*100:.1f}% predicted fatal probability. "
                         f"Extreme risk combination — Vision Zero priority scenario.")

    # ── SHAP contributors ─────────────────────────────────────────────────────
    if "shap_vals" in dir() or predict_btn or True:
        try:
            st.markdown("---")
            st.subheader("🧠 Top 3 SHAP Explainability Contributors")

            col_shap, col_shap_txt = st.columns([1, 1], gap="large")

            with col_shap:
                shap_fig = make_shap_bar(shap_vals, X)
                st.pyplot(shap_fig, use_container_width=True)
                plt.close(shap_fig)

            with col_shap_txt:
                sv_series = pd.Series(shap_vals, index=CORE_FEATURES)
                top3_feats = sv_series.abs().nlargest(3).index.tolist()

                st.markdown("**Plain-language interpretation:**")
                for i, feat in enumerate(top3_feats, 1):
                    shap_v   = sv_series[feat]
                    feat_val = X[feat].iloc[0]
                    label    = FEAT_LABELS.get(feat, feat)
                    direction = "increases" if shap_v > 0 else "reduces"
                    magnitude = "strongly" if abs(shap_v) > 0.3 else "moderately" if abs(shap_v) > 0.1 else "slightly"
                    arrow     = "🔺" if shap_v > 0 else "🔻"

                    st.markdown(
                        f"{arrow} **#{i} — {label}** (value = {feat_val:.0f})  \n"
                        f"This feature {magnitude} {direction} fatal risk "
                        f"(SHAP = {shap_v:+.3f})"
                    )
        except Exception as e:
            st.warning(f"SHAP display error: {e}")

    # ── Disclaimer ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div style="background:#1A1A2E; padding:12px 16px; border-radius:8px;
                border-left:4px solid #95A5A6; font-size:0.8rem; color:#95A5A6;">
        <b>⚠️ Research tool disclaimer:</b>
        This application is a research demonstration built on Toronto KSI data (2006–2026).
        Predictions are probabilistic and based on population-level patterns —
        they do not constitute individual traffic risk assessments.
        Not for operational or enforcement use.
        The model achieves AUC = 0.8624 on held-out test data; fatal recall = 55.8%
        at default threshold (0.5). &nbsp;|&nbsp;
        <b>Data recency note:</b> 2024–2026 data is underrepresented due to police
        reporting lag — model trained predominantly on 2006–2023 patterns.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()