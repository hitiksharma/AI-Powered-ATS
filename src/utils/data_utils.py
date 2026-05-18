
"""
data_utils.py — Utility functions for the ATS Thesis Project
Provides data loading, path management, and helper functions
used across all notebooks.
"""

import os
import json
import glob
import pandas as pd
import numpy as np
from pathlib import Path


def get_project_root():
    """Return the project root directory."""
    return os.path.expanduser("~/Desktop/thesis_final")


def load_config():
    """Load project configuration."""
    config_path = os.path.join(get_project_root(), "configs/config.json")
    with open(config_path, 'r') as f:
        return json.load(f)


def get_data_path(subfolder=""):
    """Get path to data directory or subdirectory."""
    return os.path.join(get_project_root(), "data", subfolder)


def load_resumes():
    """Load the main resume dataset."""
    resume_dir = get_data_path("raw/resumes")
    csv_files = glob.glob(os.path.join(resume_dir, "*.csv"))
    if csv_files:
        return pd.read_csv(csv_files[0])
    raise FileNotFoundError(f"No resume CSV found in {resume_dir}")


def load_job_descriptions():
    """Load job descriptions dataset."""
    jd_dir = get_data_path("raw/job_descriptions")
    csv_files = glob.glob(os.path.join(jd_dir, "*.csv"))
    if csv_files:
        return pd.read_csv(csv_files[0], low_memory=False)
    raise FileNotFoundError(f"No JD CSV found in {jd_dir}")


def load_skills_taxonomy():
    """Load the custom skills taxonomy."""
    path = get_data_path("processed/skills_taxonomy.json")
    with open(path, 'r') as f:
        return json.load(f)


def compute_text_stats(df, text_col="Resume"):
    """Compute basic text statistics for a dataframe."""
    stats = {
        "total_records": len(df),
        "avg_length": df[text_col].str.len().mean(),
        "median_length": df[text_col].str.len().median(),
        "min_length": df[text_col].str.len().min(),
        "max_length": df[text_col].str.len().max(),
        "avg_word_count": df[text_col].str.split().str.len().mean(),
        "null_count": df[text_col].isnull().sum(),
    }
    return stats


def save_figure(fig, name, dpi=300):
    """Save matplotlib figure to reports/figures/."""
    fig_dir = os.path.join(get_project_root(), "reports/figures")
    os.makedirs(fig_dir, exist_ok=True)
    path = os.path.join(fig_dir, f"{name}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"📊 Figure saved: {path}")
    return path
