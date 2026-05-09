from pathlib import Path

import pandas as pd

from src.eda_visualizations import (
    ensure_output_dir,
    plot_acclass_distribution,
    plot_categorical_distributions,
    plot_temporal_distributions,
    create_summary_statistics,
    plot_stacked_proportions,
)


def sample_eda_dataframe():
    return pd.DataFrame({
        "acclass_binary": [0, 1, 0, 1, 0],
        "invage": [25, 40, 30, 60, 35],
        "hour": [8, 8, 14, 18, 18],
        "day_of_week_name": ["Monday", "Monday", "Tuesday", "Friday", "Friday"],
        "month_name": ["January", "January", "March", "July", "July"],
        "year": [2020, 2020, 2021, 2022, 2022],
        "road_user": ["Driver", "Pedestrian", "Driver", "Cyclist", "Pedestrian"],
        "light_Daylight": [1, 0, 1, 0, 1],
        "light_Dark": [0, 1, 0, 1, 0],
        "rdsfcond_Dry": [1, 1, 0, 1, 0],
        "rdsfcond_Wet": [0, 0, 1, 0, 1],
        "traffictl_No Control": [1, 0, 1, 0, 0],
        "traffictl_Traffic Signal": [0, 1, 0, 1, 1],
        "road_class_Major Arterial": [1, 1, 0, 0, 1],
        "road_class_Local": [0, 0, 1, 1, 0],
        "accloc_At Intersection": [1, 0, 1, 0, 1],
        "accloc_Non Intersection": [0, 1, 0, 1, 0],
    })


def test_ensure_output_dir_creates_folder(tmp_path):
    output_dir = tmp_path / "story-2"

    result = ensure_output_dir(output_dir)

    assert result.exists()
    assert result.is_dir()


def test_plot_acclass_distribution_creates_png(tmp_path):
    df = sample_eda_dataframe()

    plot_acclass_distribution(df, tmp_path)

    assert (tmp_path / "task_12_acclass_distribution.png").exists()


def test_plot_categorical_distributions_creates_png_files(tmp_path):
    df = sample_eda_dataframe()

    plot_categorical_distributions(df, tmp_path)

    assert (tmp_path / "task_13_light_distribution.png").exists()
    assert (tmp_path / "task_13_rdsfcond_distribution.png").exists()
    assert (tmp_path / "task_13_traffictl_distribution.png").exists()
    assert (tmp_path / "task_13_road_class_distribution.png").exists()
    assert (tmp_path / "task_13_accloc_distribution.png").exists()


def test_plot_temporal_distributions_creates_png_files(tmp_path):
    df = sample_eda_dataframe()

    plot_temporal_distributions(df, tmp_path)

    assert (tmp_path / "task_14_hour_temporal_distribution.png").exists()
    assert (tmp_path / "task_14_day_of_week_name_temporal_distribution.png").exists()
    assert (tmp_path / "task_14_month_name_temporal_distribution.png").exists()
    assert (tmp_path / "task_14_year_temporal_distribution.png").exists()


def test_create_summary_statistics_creates_csv(tmp_path):
    df = sample_eda_dataframe()

    create_summary_statistics(df, tmp_path)

    output_file = tmp_path / "task_15_invage_summary_statistics_by_severity.csv"

    assert output_file.exists()

    summary_df = pd.read_csv(output_file)

    assert "mean" in summary_df.columns
    assert "median" in summary_df.columns
    assert "std" in summary_df.columns


def test_plot_stacked_proportions_creates_png_files(tmp_path):
    df = sample_eda_dataframe()

    plot_stacked_proportions(df, tmp_path)

    assert (tmp_path / "task_16_light_fatal_nonfatal_stacked.png").exists()
    assert (tmp_path / "task_16_rdsfcond_fatal_nonfatal_stacked.png").exists()
    assert (tmp_path / "task_16_traffictl_fatal_nonfatal_stacked.png").exists()
    assert (tmp_path / "task_16_road_user_fatal_nonfatal_stacked.png").exists()


def test_functions_do_not_crash_with_missing_optional_columns(tmp_path):
    df = pd.DataFrame({
        "acclass_binary": [0, 1, 0],
        "invage": [20, 40, 60],
    })

    plot_categorical_distributions(df, tmp_path)
    plot_temporal_distributions(df, tmp_path)
    create_summary_statistics(df, tmp_path)
    plot_stacked_proportions(df, tmp_path)

    assert (tmp_path / "task_15_invage_summary_statistics_by_severity.csv").exists()