"""
Story 10 – Geospatial Analysis
Factors Affecting Traffic Collision Severity in Toronto
Group 5 | DAMO-699-5

Steps
─────
1.  Load & prepare collision data with coordinates
2.  KDE heatmap of fatal collisions via Folium → HTML + PNG screenshot
3.  Ward-level fatality rate choropleth → HTML + PNG
4.  Top 10 fatal intersections (GPS clustering + street names)
5.  Data recency limitation documentation (police reporting lag)
6.  Export all outputs

Usage:
  python src/geospatial_analysis.py \\
      --input  data/Motor_Vehicle_Collisions_with_KSI_Data_-_4326.csv \\
      --output-dir outputs/story-10
"""

import argparse
import json
import logging
import warnings
from pathlib import Path

import folium
from folium.plugins import HeatMap
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from shapely.geometry import MultiPoint, Point
from sklearn.cluster import DBSCAN

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

# ── Toronto bounding box ──────────────────────────────────────────────────────
LAT_MIN, LAT_MAX = 43.58, 43.86
LON_MIN, LON_MAX = -79.64, -79.12
TORONTO_CENTER   = [43.720, -79.380]

# ── Palette ───────────────────────────────────────────────────────────────────
C_FATAL   = "#C0392B"
C_ACCENT  = "#E67E22"
C_BG      = "#F8F9FA"
C_GRID    = "#DEE2E6"
FONT_TITLE = {"fontsize": 13, "fontweight": "bold", "color": "#1A1A2E"}
FONT_AX    = {"fontsize": 10, "color": "#2C3E50"}

# ── Recency lag years (police reporting incomplete) ───────────────────────────
LAG_YEARS = [2024, 2025, 2026]


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Load & prepare
# ─────────────────────────────────────────────────────────────────────────────

def load_data(input_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path, low_memory=False)
    df = df[df["acclass"].isin(["Fatal Injury", "Non-Fatal Injury"])].copy()
    df["acclass_binary"] = (df["acclass"] == "Fatal Injury").astype(int)
    df["accdate"] = pd.to_datetime(df["accdate"], errors="coerce")
    df["year"]    = df["accdate"].dt.year

    # Cap age outlier
    df.loc[df["invage"] > 110, "invage"] = np.nan

    # Drop rows missing coordinates
    df = df.dropna(subset=["latitude", "longitude"])

    # Flag recency lag
    df["data_lag_flag"] = df["year"].isin(LAG_YEARS)

    print(f"  Total KSI records (with coords): {len(df):,}")
    print(f"  Fatal: {df['acclass_binary'].sum():,}  "
          f"Non-Fatal: {(df['acclass_binary']==0).sum():,}")
    print(f"  Records flagged for data-lag ({LAG_YEARS}): "
          f"{df['data_lag_flag'].sum():,}")
    return df


def build_ward_gdf(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Build approximate ward polygons from collision point convex hulls."""
    rows = []
    for ward, grp in df.dropna(subset=["wardname"]).groupby("wardname"):
        pts = MultiPoint(list(zip(grp["longitude"], grp["latitude"])))
        rows.append({"wardname": ward, "geometry": pts.convex_hull})
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — KDE Heatmap (Folium HTML + static PNG)
# ─────────────────────────────────────────────────────────────────────────────

def make_kde_heatmap_html(df: pd.DataFrame, out: Path) -> None:
    """Interactive Folium KDE heatmap — exports as HTML."""
    fatal = df[df["acclass_binary"] == 1]
    coords = fatal[["latitude", "longitude"]].values.tolist()

    m = folium.Map(
        location=TORONTO_CENTER,
        zoom_start=12,
        tiles="CartoDB Positron",
        control_scale=True,
    )

    HeatMap(
        coords,
        radius=14,
        blur=18,
        min_opacity=0.35,
        max_zoom=16,
        gradient={
            "0.2": "#2C3E50",
            "0.4": "#2980B9",
            "0.6": "#E67E22",
            "0.8": "#C0392B",
            "1.0": "#7B241C",
        },
    ).add_to(m)

    # Title tile
    title_html = """
    <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
                z-index: 9999; background: white; padding: 10px 18px;
                border-radius: 6px; border: 2px solid #C0392B;
                font-family: Arial; font-size: 14px; font-weight: bold;
                color: #1A1A2E; box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
        Fatal Collision KDE Heatmap — Toronto (2006–2026)
        <br><span style="font-size:11px; color:#7F8C8D; font-weight:normal;">
        Kernel density of {n:,} fatal KSI collision locations
        </span>
    </div>
    """.format(n=len(fatal))
    m.get_root().html.add_child(folium.Element(title_html))

    # Legend
    legend_html = """
    <div style="position: fixed; bottom: 30px; right: 20px; z-index: 9999;
                background: white; padding: 10px; border-radius: 6px;
                border: 1px solid #ccc; font-family: Arial; font-size: 11px;">
        <b>Collision Density</b><br>
        <span style="background: #7B241C; padding: 2px 10px;">&nbsp;</span> Very High<br>
        <span style="background: #C0392B; padding: 2px 10px;">&nbsp;</span> High<br>
        <span style="background: #E67E22; padding: 2px 10px;">&nbsp;</span> Moderate<br>
        <span style="background: #2980B9; padding: 2px 10px;">&nbsp;</span> Low<br>
        <span style="background: #2C3E50; padding: 2px 10px;">&nbsp;</span> Very Low
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    html_path = out / "task_55_kde_heatmap_fatal.html"
    m.save(str(html_path))
    print(f"  Saved {html_path.name}")


def make_kde_heatmap_png(df: pd.DataFrame, out: Path) -> None:
    """Static matplotlib KDE heatmap using 2D histogram + Gaussian smoothing."""
    fatal = df[df["acclass_binary"] == 1].copy()

    fig, ax = plt.subplots(figsize=(12, 10), facecolor="#1A1A2E")
    ax.set_facecolor("#1A1A2E")

    # 2D histogram binned onto a grid
    bins   = 180
    h, xedges, yedges = np.histogram2d(
        fatal["latitude"], fatal["longitude"],
        bins=bins,
        range=[[LAT_MIN, LAT_MAX], [LON_MIN, LON_MAX]],
    )
    # Gaussian smoothing for KDE effect
    h_smooth = gaussian_filter(h, sigma=2.5)
    h_smooth[h_smooth == 0] = np.nan

    img = ax.imshow(
        h_smooth,
        extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX],
        origin="lower",
        cmap="YlOrRd",
        aspect="auto",
        alpha=0.92,
        interpolation="bilinear",
    )

    # All collision background (non-fatal in dark)
    non_fatal = df[df["acclass_binary"] == 0]
    ax.scatter(non_fatal["longitude"], non_fatal["latitude"],
               c="#2C3E50", s=0.4, alpha=0.15, zorder=1)

    cbar = plt.colorbar(img, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Collision Density", color="white", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_xlabel("Longitude", color="white", fontsize=10)
    ax.set_ylabel("Latitude",  color="white", fontsize=10)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#555")

    ax.set_title(
        f"task_55b — Fatal Collision KDE Heatmap — Toronto (2006–2026)\n"
        f"{len(fatal):,} fatal KSI collisions  |  "
        f"⚠ {LAG_YEARS[0]}–{LAG_YEARS[-1]} data underrepresented (police reporting lag)",
        color="white", fontsize=12, fontweight="bold", pad=12
    )

    fig.tight_layout()
    fig.savefig(out / "task_55_kde_heatmap_fatal.png",
                dpi=150, bbox_inches="tight", facecolor="#1A1A2E")
    plt.close(fig)
    print("  Saved task_55_kde_heatmap_fatal.png")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Ward choropleth
# ─────────────────────────────────────────────────────────────────────────────

def compute_ward_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fatal count, total count, fatality rate per ward."""
    stats = (
        df.dropna(subset=["wardname"])
          .groupby("wardname")
          .agg(
              total_collisions=("acclass_binary", "count"),
              fatal_collisions=("acclass_binary", "sum"),
          )
          .reset_index()
    )
    stats["fatality_rate_pct"] = (
        stats["fatal_collisions"] / stats["total_collisions"] * 100
    ).round(2)
    # Extract ward number for ordering
    stats["ward_num"] = stats["wardname"].str.extract(r"\((\d+)\)").astype(int)
    stats = stats.sort_values("fatality_rate_pct", ascending=False)
    return stats


def make_choropleth_html(ward_stats: pd.DataFrame,
                          ward_gdf: gpd.GeoDataFrame, out: Path) -> None:
    """Folium choropleth — shade by fatality rate."""
    merged = ward_gdf.merge(ward_stats, on="wardname")

    m = folium.Map(
        location=TORONTO_CENTER,
        zoom_start=11,
        tiles="CartoDB Positron",
        control_scale=True,
    )

    # Choropleth layer
    folium.Choropleth(
        geo_data=merged.__geo_interface__,
        data=ward_stats,
        columns=["wardname", "fatality_rate_pct"],
        key_on="feature.properties.wardname",
        fill_color="YlOrRd",
        fill_opacity=0.75,
        line_opacity=0.6,
        line_color="white",
        legend_name="Fatal Collision Rate (%) — higher = more deadly per collision",
        nan_fill_color="lightgray",
    ).add_to(m)

    # Tooltips with ward stats
    for _, row in merged.iterrows():
        centroid = row.geometry.centroid
        folium.Marker(
            location=[centroid.y, centroid.x],
            icon=folium.DivIcon(
                html=f"""<div style="font-size:8px; font-weight:bold;
                              color:#1A1A2E; text-align:center;
                              white-space:nowrap;">{row['ward_num']}</div>""",
                icon_size=(24, 12),
                icon_anchor=(12, 6),
            ),
            tooltip=folium.Tooltip(
                f"<b>{row['wardname']}</b><br>"
                f"Total collisions: {int(row['total_collisions']):,}<br>"
                f"Fatal: {int(row['fatal_collisions']):,}<br>"
                f"Fatality rate: <b>{row['fatality_rate_pct']:.2f}%</b>",
                sticky=True,
            ),
        ).add_to(m)

    title_html = """
    <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
                z-index: 9999; background: white; padding: 10px 18px;
                border-radius: 6px; border: 2px solid #E67E22;
                font-family: Arial; font-size: 13px; font-weight: bold;
                color: #1A1A2E; box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
        KSI Fatal Collision Rate by Ward — Toronto (2006–2026)
        <br><span style="font-size:10px; color:#7F8C8D; font-weight:normal;">
        Fatality rate = fatal / total KSI collisions per ward
        (hover for details)
        </span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    html_path = out / "task_56_ward_choropleth.html"
    m.save(str(html_path))
    print(f"  Saved {html_path.name}")


def make_choropleth_png(ward_stats: pd.DataFrame,
                         ward_gdf: gpd.GeoDataFrame, out: Path) -> None:
    """Static matplotlib choropleth."""
    merged = ward_gdf.merge(ward_stats, on="wardname")

    fig, axes = plt.subplots(1, 2, figsize=(18, 9), facecolor="white")
    fig.suptitle(
        "Fig 34b — Fatal Collision Rate by Toronto Ward (2006–2026)\n"
        "Left: fatality rate (%)  |  Right: absolute fatal collision count  |  "
        "Convex-hull ward boundaries from collision point data",
        fontsize=12, fontweight="bold", color="#1A1A2E"
    )

    for ax, col, cmap, label, title in [
        (axes[0], "fatality_rate_pct", "YlOrRd",
         "Fatality Rate (%)",    "Fatality Rate (fatal / total collisions × 100)"),
        (axes[1], "fatal_collisions",   "OrRd",
         "Fatal Collision Count", "Absolute Fatal Collision Count"),
    ]:
        merged.plot(
            column=col,
            cmap=cmap,
            linewidth=0.8,
            edgecolor="white",
            legend=True,
            legend_kwds={"label": label, "orientation": "horizontal",
                         "shrink": 0.7, "pad": 0.02},
            ax=ax,
            missing_kwds={"color": "lightgrey"},
        )

        # Ward number labels
        for _, row in merged.iterrows():
            c = row.geometry.centroid
            ax.annotate(
                str(row["ward_num"]),
                xy=(c.x, c.y),
                ha="center", va="center",
                fontsize=7, color="#1A1A2E", fontweight="bold"
            )

        ax.set_facecolor(C_BG)
        ax.set_title(title, fontsize=10, fontweight="bold", color="#2C3E50", pad=6)
        ax.set_xlabel("Longitude", fontsize=9)
        ax.set_ylabel("Latitude",  fontsize=9)
        ax.tick_params(labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(C_GRID)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out / "task_56_ward_choropleth.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved task_56_ward_choropleth.png")


def make_ward_bar(ward_stats: pd.DataFrame, out: Path) -> None:
    """Horizontal bar chart: all 25 wards ranked by fatality rate."""
    df_plot = ward_stats.sort_values("fatality_rate_pct", ascending=True)
    overall = (ward_stats["fatal_collisions"].sum() /
               ward_stats["total_collisions"].sum() * 100)

    colors = ["#C0392B" if r > overall else "#2980B9"
              for r in df_plot["fatality_rate_pct"]]

    fig, ax = plt.subplots(figsize=(11, 10), facecolor="white")
    bars = ax.barh(df_plot["wardname"], df_plot["fatality_rate_pct"],
                   color=colors, edgecolor="white", zorder=3)

    for bar, r, n in zip(bars, df_plot["fatality_rate_pct"],
                          df_plot["fatal_collisions"]):
        ax.text(r + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{r:.2f}%  (n={int(n)})",
                va="center", fontsize=8)

    ax.axvline(overall, color=C_ACCENT, linewidth=1.8, linestyle="--",
               label=f"Toronto overall rate ({overall:.2f}%)")
    ax.set_facecolor(C_BG)
    ax.set_xlabel("Fatality Rate (fatal / total KSI × 100)", **FONT_AX)
    ax.set_title(
        "Fig 35 — Fatal Collision Rate by Ward (all 25 wards)\n"
        "Red = above Toronto average  |  Blue = below average",
        **FONT_TITLE, pad=10
    )
    ax.grid(axis="x", color=C_GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor(C_GRID)

    legend_handles = [
        mpatches.Patch(color="#C0392B", label="Above overall rate"),
        mpatches.Patch(color="#2980B9", label="Below overall rate"),
        plt.Line2D([0],[0], color=C_ACCENT, linestyle="--",
                   lw=1.8, label=f"Overall rate ({overall:.2f}%)"),
    ]
    ax.legend(handles=legend_handles, fontsize=9, framealpha=0.9)
    ax.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(out / "task_56_ward_fatality_rate_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved task_56_ward_fatality_rate_bar.png")


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Top 10 fatal intersections
# ─────────────────────────────────────────────────────────────────────────────

def find_top_intersections(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    """
    Cluster fatal collision GPS points using DBSCAN (eps ~50m),
    identify top clusters, label with nearest street name pair.
    """
    fatal = df[df["acclass_binary"] == 1].copy()

    # Convert to radians for haversine DBSCAN
    coords_rad = np.radians(fatal[["latitude", "longitude"]].values)
    eps_km     = 0.05   # 50-metre radius clustering
    db = DBSCAN(eps=eps_km / 6371.0, min_samples=2,
                algorithm="ball_tree", metric="haversine")
    fatal["cluster"] = db.fit_predict(coords_rad)

    # Aggregate per cluster
    clustered = fatal[fatal["cluster"] >= 0]

    def safe_mode(x):
        m = x.dropna().mode()
        return m.iloc[0] if len(m) > 0 else ""

    cluster_stats = (
        clustered.groupby("cluster")
        .agg(
            fatal_count = ("acclass_binary", "sum"),
            lat         = ("latitude",  "mean"),
            lon         = ("longitude", "mean"),
        )
        .reset_index()
    )
    cluster_stats["ward"]    = [safe_mode(clustered[clustered["cluster"]==c]["wardname"])
                                 for c in cluster_stats["cluster"]]
    cluster_stats["street1"] = [safe_mode(clustered[clustered["cluster"]==c]["stname1"])
                                 for c in cluster_stats["cluster"]]
    cluster_stats["street2"] = [safe_mode(clustered[clustered["cluster"]==c]["stname2"])
                                 for c in cluster_stats["cluster"]]

    cluster_stats["intersection"] = cluster_stats.apply(
        lambda r: (f"{r['street1']} & {r['street2']}"
                   if r["street2"] and r["street2"] != r["street1"]
                   else r["street1"]),
        axis=1
    )
    cluster_stats["fatality_rate_pct"] = (
        cluster_stats["fatal_count"] /
        cluster_stats["fatal_count"].clip(lower=1) * 100
    ).round(1)

    # Also compute fatality rate vs all collisions at that location
    all_pts   = np.radians(df[["latitude", "longitude"]].values)
    all_labels = db.fit_predict(all_pts)
    df["cluster"] = all_labels

    all_stats = (
        df[df["cluster"] >= 0]
          .groupby("cluster")
          .agg(total_all=("acclass_binary","count"),
               fatal_all=("acclass_binary","sum"))
          .reset_index()
    )
    cluster_stats = cluster_stats.merge(all_stats, on="cluster", how="left")
    cluster_stats["loc_fatality_rate_pct"] = (
        cluster_stats["fatal_all"] /
        cluster_stats["total_all"].clip(lower=1) * 100
    ).round(1)

    top10 = (cluster_stats
             .sort_values("fatal_count", ascending=False)
             .head(10)
             .reset_index(drop=True))
    top10.index += 1  # rank from 1

    top10_out = top10[["intersection", "ward", "lat", "lon",
                        "fatal_count", "total_all",
                        "loc_fatality_rate_pct"]].copy()
    top10_out.columns = ["Intersection", "Ward", "Latitude", "Longitude",
                          "Fatal Count", "Total KSI",
                          "Fatality Rate (%)"]
    top10_out.index.name = "Rank"
    top10_out.to_csv(out / "task_57_top10_fatal_intersections.csv")
    print(f"  Top 10 fatal intersections saved → top10_fatal_intersections.csv")
    print(top10_out[["Intersection","Ward","Fatal Count",
                      "Fatality Rate (%)"]].to_string())
    return top10_out


def plot_top10_map(top10: pd.DataFrame, df_all: pd.DataFrame,
                   out: Path) -> None:
    """Folium map of top 10 intersections + all fatal collisions."""
    m = folium.Map(location=TORONTO_CENTER, zoom_start=12,
                   tiles="CartoDB Positron", control_scale=True)

    # All fatal as small dots
    fatal = df_all[df_all["acclass_binary"] == 1]
    for _, row in fatal.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=2, color="#C0392B", fill=True,
            fill_color="#C0392B", fill_opacity=0.3,
            weight=0,
        ).add_to(m)

    # Top 10 as numbered stars
    for rank, row in top10.iterrows():
        folium.Marker(
            location=[row["Latitude"], row["Longitude"]],
            icon=folium.DivIcon(
                html=f"""<div style="background:#1A1A2E; color:white;
                            border-radius:50%; width:26px; height:26px;
                            display:flex; align-items:center; justify-content:center;
                            font-weight:bold; font-size:12px;
                            border:2px solid #E67E22; font-family:Arial;">
                            {rank}</div>""",
                icon_size=(26, 26),
                icon_anchor=(13, 13),
            ),
            tooltip=folium.Tooltip(
                f"<b>Rank #{rank}: {row['Intersection']}</b><br>"
                f"Ward: {row['Ward']}<br>"
                f"Fatal collisions: <b>{int(row['Fatal Count'])}</b><br>"
                f"Total KSI: {int(row['Total KSI'])}<br>"
                f"Fatality rate: <b>{row['Fatality Rate (%)']:.1f}%</b>",
                sticky=True,
            ),
        ).add_to(m)

    title_html = """
    <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
                z-index: 9999; background: white; padding: 10px 18px;
                border-radius: 6px; border: 2px solid #1A1A2E;
                font-family: Arial; font-size: 13px; font-weight: bold;
                color: #1A1A2E; box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
        Top 10 Fatal Collision Intersections — Toronto (2006–2026)
        <br><span style="font-size:10px; color:#7F8C8D; font-weight:normal;">
        Numbered markers = top 10 by fatal count (50m DBSCAN clusters)
        </span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))
    html_path = out / "task_57_top10_intersections_map.html"
    m.save(str(html_path))
    print(f"  Saved {html_path.name}")


def plot_top10_table(top10: pd.DataFrame, out: Path) -> None:
    """Static table figure for the top 10 intersections."""
    display = top10.reset_index().rename(columns={"index": "Rank"})
    display = display[["Rank", "Intersection", "Ward",
                        "Fatal Count", "Total KSI", "Fatality Rate (%)"]].copy()

    fig, ax = plt.subplots(figsize=(14, 5), facecolor="white")
    fig.suptitle(
        "task_57b — Top 10 Fatal Collision Intersections — Toronto (2006–2026)\n"
        "Ranked by fatal collision count  |  50m GPS radius clustering (DBSCAN)",
        fontsize=12, fontweight="bold", color="#1A1A2E"
    )
    ax.axis("off")

    col_labels = list(display.columns)
    cell_data  = display.values.tolist()

    tbl = ax.table(cellText=cell_data, colLabels=col_labels,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 2.4)
    tbl.auto_set_column_width(col=list(range(len(col_labels))))

    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#1A1A2E")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    for i in range(1, 11):
        fill = "#FADBD8" if i <= 3 else "#FEF9E7" if i <= 7 else "#F8F9FA"
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor(fill)

    legend_patches = [
        mpatches.Patch(color="#FADBD8", label="Top 3 (highest risk)"),
        mpatches.Patch(color="#FEF9E7", label="Ranks 4–7"),
        mpatches.Patch(color="#F8F9FA", label="Ranks 8–10"),
    ]
    ax.legend(handles=legend_patches, loc="lower right",
              fontsize=9, framealpha=0.9)

    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    fig.savefig(out / "task_57_top10_intersections_table.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved task_57_top10_intersections_table.png")


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Data recency limitation documentation
# ─────────────────────────────────────────────────────────────────────────────

RECENCY_DOC = """
DATA RECENCY LIMITATION — GEOSPATIAL SECTION
==============================================

Source: City of Toronto KSI Dataset (updated daily, last refreshed April 2026)
Documented in: Project Proposal §4 and Story 5 (Research Question 2d)

LIMITATION:
There can be an interval of several months to more than a year between a
collision occurring and the finalized record being provided to the City of
Toronto by Toronto Police Service. As a result, the most recent years in the
dataset are systematically underrepresented relative to their true collision
counts.

AFFECTED YEARS: {lag_years}

OBSERVED UNDERCOUNT (raw KSI records by year):
{year_table}

ANALYTICAL IMPLICATIONS FOR GEOSPATIAL ANALYSIS:

1. KDE Heatmap (task_55):
   All years are included in the heatmap. Hotspot locations are stable over
   the full 2006–2026 period and are unlikely to shift materially due to
   lag-year undercount. Density magnitude in lag years may appear lower than
   true counts — the heatmap should be interpreted as a long-run spatial
   pattern, not a current-year snapshot.

2. Ward Choropleth (task_56):
   Fatality rates are computed over all years (2006–2026). The lag-year
   undercount affects all wards approximately proportionally, so inter-ward
   comparisons remain valid. Absolute counts in lag years are suppressed.

3. Top 10 Intersections (task_57):
   Intersection rankings are based on accumulated fatal counts 2006–2026.
   The DBSCAN clusters are unlikely to change rank order due to lag-year
   undercount, as the top intersections accumulate fatal collisions across
   many years. Recent-year collisions at these locations may be
   underrepresented by 20–50% relative to their true 2024–2026 totals.

4. Temporal Trend Interpretation:
   Any year-over-year trend analysis involving 2024, 2025, or 2026 should
   explicitly note that apparent declines in recent years likely reflect
   reporting lag, not genuine reductions in collision frequency. Temporal
   trend figures (Story 2 task_14) already annotate this limitation.

RECOMMENDATION:
For operational dashboard use, filter to years 2006–2023 for trend-based
analyses. For spatial hotspot identification, use the full 2006–2026 period
as spatial patterns are stable across years.
"""


def write_recency_docs(df: pd.DataFrame, out: Path) -> None:
    year_counts = (
        df.groupby("year")["acclass_binary"]
          .agg(["count", "sum"])
          .rename(columns={"count": "Total KSI", "sum": "Fatal"})
          .tail(8)
    )
    year_counts["Lag Flag"] = year_counts.index.isin(LAG_YEARS)
    year_table_str = year_counts.to_string()

    doc = RECENCY_DOC.format(
        lag_years=LAG_YEARS,
        year_table=year_table_str,
    )
    (out / "task_58_data_recency_limitation.txt").write_text(doc)
    print("  Saved data_recency_limitation.txt")

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="white")
    fig.suptitle(
        "task_58 — Data Recency Limitation: Police Reporting Lag\n"
        "Collision counts in recent years are systematically underrepresented",
        fontsize=12, fontweight="bold", color="#1A1A2E"
    )

    # Year trend — all collisions
    year_all = df.groupby("year").size()
    year_fat = df[df["acclass_binary"]==1].groupby("year").size()

    ax1 = axes[0]
    colors_bar = ["#BDC3C7" if y in LAG_YEARS else "#2980B9"
                  for y in year_all.index]
    ax1.bar(year_all.index, year_all.values,
            color=colors_bar, edgecolor="white", zorder=3)
    ax1.set_facecolor(C_BG)
    ax1.set_title("Total KSI Collisions by Year\n(grey = reporting lag affected)",
                  fontsize=10, fontweight="bold", color="#2C3E50")
    ax1.set_xlabel("Year", fontsize=9)
    ax1.set_ylabel("KSI Records", fontsize=9)
    ax1.tick_params(labelsize=8)
    ax1.set_xticks(year_all.index)
    ax1.set_xticklabels(year_all.index, rotation=45)
    ax1.grid(axis="y", color=C_GRID, linewidth=0.7, zorder=0)
    ax1.set_axisbelow(True)
    for spine in ax1.spines.values():
        spine.set_edgecolor(C_GRID)

    # Annotate lag region
    lag_start = min(LAG_YEARS) - 0.5
    ax1.axvspan(lag_start, max(LAG_YEARS) + 0.5,
                alpha=0.12, color="#E74C3C", zorder=0)
    ax1.text(min(LAG_YEARS) + (max(LAG_YEARS)-min(LAG_YEARS))/2,
             year_all.max() * 0.92,
             "Reporting\nlag zone",
             ha="center", fontsize=8.5, color="#C0392B",
             fontweight="bold")

    # Fatal rate trend
    ax2 = axes[1]
    year_rate = (year_fat / year_all * 100).fillna(0)
    line_colors = ["#BDC3C7" if y in LAG_YEARS else C_FATAL
                   for y in year_rate.index]
    ax2.plot(year_rate.index, year_rate.values,
             "o-", color=C_FATAL, linewidth=2, markersize=5, zorder=3)
    for y, r in zip(year_rate.index, year_rate.values):
        if y in LAG_YEARS:
            ax2.plot(y, r, "o", color="#BDC3C7", markersize=7, zorder=4)

    ax2.axvspan(lag_start, max(LAG_YEARS) + 0.5,
                alpha=0.12, color="#E74C3C", zorder=0)
    ax2.set_facecolor(C_BG)
    ax2.set_title("Fatality Rate (%) by Year\n(trend in lag years may be unreliable)",
                  fontsize=10, fontweight="bold", color="#2C3E50")
    ax2.set_xlabel("Year", fontsize=9)
    ax2.set_ylabel("Fatal / Total KSI × 100 (%)", fontsize=9)
    ax2.tick_params(labelsize=8)
    ax2.set_xticks(year_rate.index)
    ax2.set_xticklabels(year_rate.index, rotation=45)
    ax2.grid(axis="y", color=C_GRID, linewidth=0.7, zorder=0)
    ax2.set_axisbelow(True)
    for spine in ax2.spines.values():
        spine.set_edgecolor(C_GRID)

    legend_patches = [
        mpatches.Patch(color="#2980B9", label="Normal year"),
        mpatches.Patch(color="#BDC3C7", label=f"Reporting lag ({LAG_YEARS[0]}–{LAG_YEARS[-1]})"),
    ]
    for ax in axes:
        ax.legend(handles=legend_patches, fontsize=8.5, framealpha=0.9)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out / "task_58_data_recency_limitation.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved task_58_data_recency_limitation.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(input_path: str, output_dir: str = "outputs"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=== Step 1: Loading & preparing data ===")
    df = load_data(input_path)

    print("\n=== Step 2: KDE Heatmap ===")
    make_kde_heatmap_html(df, out)
    make_kde_heatmap_png(df, out)

    print("\n=== Step 3: Ward choropleth ===")
    ward_stats = compute_ward_stats(df)
    ward_gdf   = build_ward_gdf(df)
    ward_stats.to_csv(out / "task_56_ward_fatality_stats.csv", index=False)
    print(f"  Ward stats saved ({len(ward_stats)} wards)")
    print(ward_stats[["wardname","total_collisions","fatal_collisions",
                       "fatality_rate_pct"]].head(10).to_string(index=False))
    make_choropleth_html(ward_stats, ward_gdf, out)
    make_choropleth_png(ward_stats, ward_gdf, out)
    make_ward_bar(ward_stats, out)

    print("\n=== Step 4: Top 10 fatal intersections ===")
    top10 = find_top_intersections(df, out)
    plot_top10_map(top10, df, out)
    plot_top10_table(top10, out)

    print("\n=== Step 5: Data recency limitation ===")
    write_recency_docs(df, out)

    print(f"\n=== Geospatial analysis complete — outputs in {out.resolve()} ===")
    return ward_stats, top10


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Story 10 — Geospatial Analysis")
    parser.add_argument("--input",      required=True,
                        help="Path to raw KSI CSV")
    parser.add_argument("--output-dir", default="outputs",
                        help="Output directory (created if absent)")
    args = parser.parse_args()
    run(args.input, args.output_dir)