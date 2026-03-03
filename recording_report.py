from __future__ import annotations # Lets Python treat type hints as postponed strings internally, no "" needed before defining
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

# Import DataJoint table classes
from neuropixel_data_schema import (
    AllTrialSpikeCount,
    Recording,
    Trial,
    TrialUnitSpikeCount,
    Unit,
    AllUnitMap,
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

# Unit numbers by recording and region
def unit_counts_by_region(recording_id: str | None = None, include_all: bool = True) -> pd.DataFrame:
    u = _to_df(Unit)
    if u.empty:
        return u

    base = u[u["brain_region"] != "all"].copy()
    if recording_id is not None:
        base = base[base["recording_id"] == recording_id]

    base_counts = (
        base.groupby(["recording_id", "brain_region"], as_index=False)
        .size()
        .rename(columns={"size": "n_units"})
    )

    if not include_all:
        return base_counts.sort_values(["recording_id", "brain_region"])

    a = _to_df(AllUnitMap)
    if recording_id is not None:
        a = a[a["recording_id"] == recording_id]

    all_counts = (
        a.groupby(["recording_id"], as_index=False)
        .size()
        .rename(columns={"size": "n_units"})
    )
    all_counts["brain_region"] = "all"

    return pd.concat([base_counts, all_counts], ignore_index=True).sort_values(
        ["recording_id", "brain_region"]
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

# Same as above but in Hz, therefore norm. by trial length
def trial_rate_totals(recording_id: str | None = None, include_all: bool = True, by_trial_type: bool = True) -> pd.DataFrame:
    totals = trial_totals(recording_id=recording_id, include_all=include_all, by_trial_type=by_trial_type)

    meta = pd.DataFrame(Trial.fetch("recording_id", "trial_idx", "trial_type", "trial_length", "bin_size_sec", as_dict=True))
    if recording_id is not None:
        meta = meta[meta["recording_id"] == recording_id]

    out = totals.drop(columns=["trial_type"], errors="ignore").merge(
        meta, on=["recording_id", "trial_idx"], how="left"
    )
    if by_trial_type:
        out["trial_type"] = out["trial_type"].fillna("unknown")

    out["trial_duration_sec"] = out["trial_length"] * out["bin_size_sec"]
    out["spikes_per_sec"] = out["trial_total_spikes"] / out["trial_duration_sec"].replace(0, pd.NA)
    return out

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

# Plot spike amounts per trial, split for trial types
from matplotlib.lines import Line2D

def plot_trial_totals(
    recording_id: str | None = None,
    include_all: bool = True,
    trial_types: tuple[str, ...] = ("trial", "intertrial"),
):
    totals = trial_totals(
        recording_id=recording_id,
        include_all=include_all,
        by_trial_type=True,
    )
    if totals.empty:
        raise ValueError("No trial totals found.")

    if recording_id is not None:
        totals = totals[totals["recording_id"] == recording_id]

    if trial_types:
        totals = totals[totals["trial_type"].isin(trial_types)]

    if totals.empty:
        raise ValueError("No rows left after trial_type filtering.")

    g = sns.relplot(
        data=totals,
        x="trial_idx",
        y="trial_total_spikes",
        hue="brain_region",
        col="trial_type",
        row="recording_id" if recording_id is None else None,
        kind="line",
        facet_kws={"sharey": False, "sharex": False},
        height=3.6,
        aspect=1.3,
    )

    # derive perturbation index per recording from Trial metadata
    meta = pd.DataFrame(Trial.fetch("recording_id", "trial_type", "idx_sol_on", as_dict=True))
    meta = meta[(meta["trial_type"] == "trial") & (meta["idx_sol_on"].notna())]
    pert_idx = meta.groupby("recording_id")["idx_sol_on"].median().to_dict()

    # draw red line only in 'trial' facets
    if recording_id is None:
        for key, ax in g.axes_dict.items():
            if isinstance(key, tuple):
                rid, ttype = key[0], key[1]
            else:
                rid, ttype = None, key
            if ttype == "trial":
                x = pert_idx.get(rid)
                if pd.notna(x):
                    ax.axvline(float(x), color="red", linestyle="--", linewidth=1.5)
                    ax.text(float(x), ax.get_ylim()[1] * 0.95, "perturbation", color="red", ha="right", va="top")
    else:
        x = pert_idx.get(recording_id)
        if pd.notna(x):
            for key, ax in g.axes_dict.items():
                ttype = key[-1] if isinstance(key, tuple) else key
                if ttype == "trial":
                    ax.axvline(float(x), color="red", linestyle="--", linewidth=1.5)
                    ax.text(float(x), ax.get_ylim()[1] * 0.95, "perturbation", color="red", ha="right", va="top")

    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.set_axis_labels("trial_idx", "total spikes")

    # combine seaborn legend + custom perturbation handle
    handles, labels = [], []
    if g._legend is not None:
        try:
            handles = list(g._legend.legend_handles)  # matplotlib >= 3.7
        except AttributeError:
            handles = g._legend.legendHandles         # older fallback
        labels = [t.get_text() for t in g._legend.texts]
        g._legend.remove()

    pert_handle = Line2D([0], [0], color="red", linestyle="--", linewidth=1.5, label="perturbation")
    handles.append(pert_handle)
    labels.append("perturbation")

    g.fig.legend(
        handles=handles,
        labels=labels,
        title="brain_region / marker",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )

    sup = recording_id if recording_id else "All recordings"
    g.fig.suptitle(f"Recording: {sup} | totals by trial type", y=1.02)

    # g.fig.subplots_adjust(right=0.80)
    # g.fig.tight_layout(rect=[0, 0, 0.80, 0.96])
    g.fig.tight_layout()
    return g

def plot_unit_sparsity_and_rate(recording_id: str):
    ua = unit_activity(recording_id=recording_id, include_all=True).copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Sparsity: box + points
    sns.boxplot(
        data=ua,
        x="brain_region",
        y="active_fraction",
        ax=axes[0],
        showfliers=False,
    )
    sns.stripplot(
        data=ua,
        x="brain_region",
        y="active_fraction",
        ax=axes[0],
        color="black",
        alpha=0.25,
        size=2,
    )
    axes[0].set_title(f"Unit Sparsity (Active Fraction)\n{recording_id}")
    # axes[0].set_ylim(0, 1)

    # Rate: box plot on log scale
    sns.boxplot(
        data=ua,
        x="brain_region",
        y="avg_spikes_per_trial",
        ax=axes[1],
        showfliers=False,
    )
    axes[1].set_title(f"Unit Mean Spikes/Trial\n{recording_id}")
    axes[1].set_yscale("log")

    for ax in axes:
        ax.tick_params(axis="x", rotation=30)

    fig.tight_layout(pad=0.8, w_pad=0.8, h_pad=0.8)
    return fig

def plot_summary_bars(recording_id: str, include_all: bool = True, log_y: bool = False):
    """
    Two-panel summary:
    1) n_units by region
    2) mean_spikes by region with std_spikes error bars
    """
    summary = build_recording_report_table(recording_id=recording_id, include_all=include_all).copy()
    if summary.empty:
        raise ValueError(f"No summary data for recording_id={recording_id}")

    summary = summary.sort_values("brain_region").reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: unit counts
    sns.barplot(data=summary, x="brain_region", y="n_units", ax=axes[0], palette="deep")
    axes[0].set_title("Units by Region")
    axes[0].set_xlabel("Region")
    axes[0].set_ylabel("n_units")
    axes[0].tick_params(axis="x", rotation=30)

    # Plot 2: mean spikes with std error bars
    sns.barplot(data=summary, x="brain_region", y="mean_spikes", ax=axes[1], palette="deep", errorbar=None)
    axes[1].errorbar(
        x=range(len(summary)),
        y=summary["mean_spikes"],
        yerr=summary["std_spikes"],
        fmt="none",
        ecolor="black",
        elinewidth=1.5,
        capsize=4,
    )
    axes[1].set_title("Mean Spikes by Region (+/- SD)")
    axes[1].set_xlabel("Region")
    axes[1].set_ylabel("mean_spikes")
    axes[1].tick_params(axis="x", rotation=30)

    if log_y:
        axes[1].set_yscale("log")

    fig.suptitle(f"Recording: {recording_id}")
    fig.tight_layout()
    return fig



# NEW: cleaner reporting table
def build_recording_report_table(recording_id: str, include_all: bool = True) -> pd.DataFrame:
    units = unit_counts_by_region(recording_id=recording_id, include_all=include_all)
    stats = spike_stats_by_region(recording_id=recording_id, include_all=include_all)
    zeros = zero_fraction_by_region(recording_id=recording_id, include_all=include_all)

    meta = recording_metadata(recording_id=recording_id)[
        ["recording_id", "animal", "session_label", "part_index", "n_trials"]
    ]

    out = (
        units.merge(stats, on=["recording_id", "brain_region"], how="outer")
             .merge(zeros, on=["recording_id", "brain_region"], how="outer")
             .merge(meta, on="recording_id", how="left")
             .sort_values(["recording_id", "brain_region"])
             .reset_index(drop=True)
    )
    return out

# OLD: build the whole report and make csv file 
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