from __future__ import annotations # Lets Python treat type hints as postponed strings internally, no "" needed before defining
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Import DataJoint table classes
from neuropixel_data_schema import (
    AllTrialSpikeCount,
    Recording,
    Trial,
    TrialUnitSpikeCount,
    Unit,
)

# Set style for plotting
sns.set_theme(style="whitegrid", context="talk") # white grid background, larger “talk” scale fonts/lines.

# Converts a DataJoint relation to pandas.
def _to_df(rel) -> pd.DataFrame:
    df = rel.to_pandas()
    if isinstance(df.index, pd.MultiIndex) or df.index.name is not None:
        df = df.reset_index()
    return df

# Returns recording data, if no key then all recordings
def recording_metadata(recording_id: str | None = None) -> pd.DataFrame:
    rel = Recording if recording_id is None else (Recording & {"recording_id": recording_id})
    return _to_df(rel)

# Aggregates trials per recording, with new n_trials_loaded column
def trial_counts(recording_id: str | None = None) -> pd.DataFrame:
    rel = Recording.aggr(Trial, n_trials_loaded="count(*)")
    if recording_id is not None:
        rel = rel & {"recording_id": recording_id}
    return _to_df(rel)

# 
def unit_counts_by_region(recording_id: str | None = None, include_all: bool = True) -> pd.DataFrame:
    unit_df = _to_df(Unit) # Pulls all Unit rows
    if unit_df.empty: # if nothing there
        return unit_df
    
    if not include_all: # dont include 'all' column
        unit_df = unit_df[unit_df["brain_region"] != "all"]

    if recording_id is not None: # optional single-recording filter
        unit_df = unit_df[unit_df["recording_id"] == recording_id]
    
    return (
        unit_df.groupby(["recording_id", "brain_region"], as_index=False) # Groups by recording + region
        .size() # ordering
        .rename(columns={"size": "n_units"}) # cleaner
        .sort_values(["recording_id", "brain_region"])
    )

# Joins trial-unit spike counts with trial metadata, so we know what mouse, etc. was the spike data from :)
def _base_spike_df(recording_id: str | None = None) -> pd.DataFrame:
    rel = TrialUnitSpikeCount * Trial # * join in DJ

    if recording_id is not None: # if 1 recording wanted
        rel = rel & {"recording_id": recording_id}

    df = _to_df(rel) # make df again
    if df.empty:
        return df
    return df[df["brain_region"] != "all"].copy() # NOTE: remove raw 'all' rows from imported data

# Uses computed all table (done in 'neuropixel_populate_computed) + trial join
def _computed_all_df(recording_id: str | None = None) -> pd.DataFrame:
    rel = AllTrialSpikeCount * Trial

    if recording_id is not None:
        rel = rel & {"recording_id": recording_id}
    df = _to_df(rel)
    if df.empty:
        return pd.DataFrame(columns=["recording_id", "trial_idx", "brain_region", "unit_idx", "n_spikes"]) # empty thing
    
    # standardizes computed schema to match base spike schema + renames key column and sets brain_region="all"
    df = df.rename(columns={"all_unit_idx": "unit_idx"})
    df["brain_region"] = "all"
    return df[["recording_id", "trial_idx", "brain_region", "unit_idx", "n_spikes"]].copy()

# Build a unified spike data table
def spike_counts(recording_id: str | None = None, include_all: bool = True) -> pd.DataFrame:
    base_df = _base_spike_df(recording_id=recording_id) # get raw columns
    if not include_all:
        return base_df
    return pd.concat([base_df, _computed_all_df(recording_id=recording_id)], ignore_index=True) #add 'all' spike data

# Spike statistics, spike counts, mean, median, std for each brain region 
def spike_stats_by_region(recording_id: str | None = None, include_all: bool = True) -> pd.DataFrame:
    df = spike_counts(recording_id=recording_id, include_all=include_all)

    if df.empty:
        return df
    
    return (
        df.groupby(["recording_id", "brain_region"], as_index=False)["n_spikes"]
        .agg(mean_spikes="mean", median_spikes="median", std_spikes="std")
        .sort_values(["recording_id", "brain_region"])
    )

# Creates is_zero indicator (1 if zero spikes), important so stats dont fail (Mean of indicator = fraction of zero rows)
# Essentially asking: for each region, what fraction of trial-unit rows had zero spikes?
def zero_fraction_by_region(recording_id: str | None = None, include_all: bool = True) -> pd.DataFrame:
    df = spike_counts(recording_id=recording_id, include_all=include_all)
    if df.empty:
        return df
    return (
        df.assign(is_zero=df["n_spikes"].eq(0).astype(float))
        .groupby(["recording_id", "brain_region"], as_index=False)["is_zero"]
        .mean()
        .rename(columns={"is_zero": "zero_fraction"})
        .sort_values(["recording_id", "brain_region"])
    )

# Trial spike numbers
def trial_totals(recording_id: str | None = None, include_all: bool = True, by_trial_type: bool = False) -> pd.DataFrame:
    df = spike_counts(recording_id=recording_id, include_all=include_all)
    if df.empty:
        return df

    trial_meta = pd.DataFrame(Trial.fetch("recording_id", "trial_idx", "trial_type", as_dict=True))
    if trial_meta.empty:
        trial_meta = pd.DataFrame(columns=["recording_id", "trial_idx", "trial_type"])

    # avoid trial_type_x / trial_type_y collisions
    df = df.drop(columns=["trial_type"], errors="ignore")
    df = df.merge(trial_meta, on=["recording_id", "trial_idx"], how="left")
    df["trial_type"] = df.get("trial_type", pd.Series(index=df.index, dtype=object)).fillna("unknown")

    group_cols = ["recording_id", "brain_region", "trial_idx"]
    if by_trial_type:
        group_cols.append("trial_type")

    return (
        df.groupby(group_cols, as_index=False)["n_spikes"]
        .sum()
        .rename(columns={"n_spikes": "trial_total_spikes"})
    )


def outlier_trials(recording_id: str | None = None, include_all: bool = True, k: float = 1.5) -> pd.DataFrame:
    totals = trial_totals(recording_id=recording_id, include_all=include_all)
    if totals.empty:
        return totals
    
    # quartile calcs for the spike numbers
    g = totals.groupby(["recording_id", "brain_region"])["trial_total_spikes"]
    # computes outlier bounds per group using Tukey/IQR method
    q1 = g.transform(lambda s: s.quantile(0.25))
    q3 = g.transform(lambda s: s.quantile(0.75))
    iqr = q3 - q1 # middle-50% spread

    # NOTE: k controls strictness: smaller k (e.g., 1.0) => tighter bounds => more outliers; 
    # larger k (e.g., 2.0) => wider bounds => fewer outliers; 
    # common default is 1.5

    # low/high are outlier cutoffs
    low = q1 - k * iqr
    high = q3 + k * iqr
    # keeps only values outside low and high
    flagged = totals[(totals["trial_total_spikes"] < low) | (totals["trial_total_spikes"] > high)]
    return flagged.sort_values(["recording_id", "brain_region", "trial_idx"])

def unit_activity(recording_id: str | None = None, include_all: bool = True) -> pd.DataFrame:
    df = spike_counts(recording_id=recording_id, include_all=include_all)
    if df.empty:
        return df
    
    return (
        df.assign(is_active=df["n_spikes"].gt(0).astype(float)) # 1 if unit fired on a trial
        .groupby(["recording_id", "brain_region", "unit_idx"], as_index=False)
        .agg(avg_spikes_per_trial=("n_spikes", "mean"), # mean spikes for that unit across trials
             active_fraction=("is_active", "mean")) # fraction of trials where unit fired
        .sort_values(["recording_id", "brain_region", "avg_spikes_per_trial"], ascending=[True, True, False])
    )

# Gets unit activity table and shows n top and n bottom firing units
def top_bottom_units(recording_id: str | None = None, include_all: bool = True, n: int = 10):
    ua = unit_activity(recording_id=recording_id, include_all=include_all)
    if ua.empty:
        return ua, ua
    
    top = (
        ua.sort_values("avg_spikes_per_trial", ascending=False)
        .groupby(["recording_id", "brain_region"], as_index=False)
        .head(n)
        .reset_index(drop=True)
    )

    bottom = (
        ua.sort_values("avg_spikes_per_trial", ascending=True)
        .groupby(["recording_id", "brain_region"], as_index=False)
        .head(n)
        .reset_index(drop=True)
    )

    return top, bottom

# As one recoding is on two files _0 and _1 we would like to see if the data is consistent or differs stats-wise
def part_consistency_summary(include_all: bool = False) -> pd.DataFrame:
    stats = spike_stats_by_region(include_all=include_all)
    meta = recording_metadata()[["recording_id", "part_index", "session_label", "animal"]]
    if stats.empty:
        return stats
    
    merged = stats.merge(meta, on="recording_id", how="left")
    return merged[
        ["animal", "session_label", "part_index", "recording_id", "brain_region", "median_spikes", "mean_spikes"]
    ].sort_values(["animal", "session_label", "brain_region", "part_index"])

# Plotting - spike counts per trial
def plot_trial_totals(recording_id: str | None = None, include_all: bool = True, by_trial_type: bool = False):
    totals = trial_totals(recording_id=recording_id, include_all=include_all, by_trial_type=by_trial_type)
    if totals.empty:
        raise ValueError("No trial totals found.")

    if recording_id is not None:
        totals = totals[totals["recording_id"] == recording_id]

    if by_trial_type:
        g = sns.relplot(
            data=totals,
            x="trial_idx",
            y="trial_total_spikes",
            hue="brain_region",
            col="trial_type",
            row="recording_id" if recording_id is None else None,
            kind="line",
            facet_kws={"sharey": False, "sharex": True},
            height=4,
            aspect=1.4,
        )
        title = f"Trial Total Spikes by Trial Type ({recording_id})" if recording_id else "Trial Total Spikes by Trial Type (All Recordings)"
        g.fig.suptitle(title, y=1.02)
        return g
    else:
        g = sns.relplot(
            data=totals,
            x="trial_idx",
            y="trial_total_spikes",
            hue="brain_region",
            col="recording_id" if recording_id is None else None,
            kind="line",
            height=4,
            aspect=1.6,
        )
        title = f"Trial Total Spikes ({recording_id})" if recording_id else "Trial Total Spikes (All Recordings)"
        g.fig.suptitle(title, y=1.02)
        return g
    
def trial_rate_totals(recording_id: str | None = None, include_all: bool = True, by_trial_type: bool = True) -> pd.DataFrame:
    totals = trial_totals(recording_id=recording_id, include_all=include_all, by_trial_type=by_trial_type)
    trial_meta = _to_df(Trial)[["recording_id", "trial_idx", "trial_length", "bin_size_sec", "trial_type"]]
    out = totals.merge(trial_meta, on=["recording_id", "trial_idx"] + (["trial_type"] if by_trial_type else []), how="left")
    out["trial_duration_sec"] = out["trial_length"] * out["bin_size_sec"]
    out["spikes_per_sec"] = out["trial_total_spikes"] / out["trial_duration_sec"].replace(0, pd.NA)
    return out

# Plotting - per-region boxplot of spike count distribution
def plot_spike_distributions(recording_id: str, include_all: bool = True):
    df = spike_counts(recording_id=recording_id, include_all=include_all)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df, x="brain_region", y="n_spikes", ax=ax, showfliers=True) # put in showfliers=False for clearer plot
    ax.set_title(f"Spike Count Distribution by Region: {recording_id}")
    ax.set_xlabel("Region")
    ax.set_ylabel("n_spikes (per trial-unit row)")
    fig.tight_layout()
    return fig

# build the whole report and make csv file 
def build_recording_report(recording_id: str, include_all: bool = True, top_n: int = 10):
    top, bottom = top_bottom_units(recording_id=recording_id, include_all=include_all, n=top_n)

    return {
        "recording_metadata": recording_metadata(recording_id=recording_id),
        "trial_counts": trial_counts(recording_id=recording_id),
        "unit_counts_by_region": unit_counts_by_region(recording_id=recording_id, include_all=include_all),
        "spike_stats_by_region": spike_stats_by_region(recording_id=recording_id, include_all=include_all),
        "zero_fraction_by_region": zero_fraction_by_region(recording_id=recording_id, include_all=include_all),
        "trial_totals": trial_totals(recording_id=recording_id, include_all=include_all),
        "outlier_trials": outlier_trials(recording_id=recording_id, include_all=include_all),
        "unit_activity": unit_activity(recording_id=recording_id, include_all=include_all),
        "top_units": top,
        "bottom_units": bottom,
    }

# Saves report in new dir if needed
def save_report_tables(report: dict[str, pd.DataFrame], output_dir: str) -> None:
    from pathlib import Path
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    for name, df in report.items():
        df.to_csv(outdir / f"{name}.csv", index=False)