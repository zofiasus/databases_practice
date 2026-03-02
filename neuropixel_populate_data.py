import argparse
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.io as sio

from neuropixel_data_schema import Recording, Trial, TrialUnitSpikeCount, Unit

# Maps your pyaldata fields to a unified region name.
REGION_FIELDS = [
    ("Thal", "Thal_spikes", "Thal_unit_guide", "Thal_KSLabel"),
    ("CP", "CP_spikes", "CP_unit_guide", "CP_KSLabel"),
    ("MOp", "MOp_spikes", "MOp_unit_guide", "MOp_KSLabel"),
    ("SSp_ll", "SSp_ll_spikes", "SSp_ll_unit_guide", "SSp_ll_KSLabel"),
    ("all", "all_spikes", "all_unit_guide", "all_KSLabel"),
]


def parse_filename(path: Path) -> dict:
    """
    Expected:
      M046_2024_12_19_13_30_pyaldata_0.mat
    """
    m = re.match(
        r"(?P<animal>[^_]+)_(?P<date>\d{4}_\d{2}_\d{2})_(?P<time>\d{2}_\d{2})_pyaldata_(?P<part>\d+)\.mat$",
        path.name,
    )
    if not m:
        raise ValueError(f"Unexpected filename format: {path.name}")

    session_datetime = datetime.strptime(
        f"{m.group('date')}_{m.group('time')}", "%Y_%m_%d_%H_%M"
    )
    return {
        "animal": m.group("animal"),
        "session_label": f"{m.group('date')}_{m.group('time')}",
        "session_datetime": session_datetime,
        "part_index": int(m.group("part")), # if data of 1 recording stored in two or more files
    }


def load_trials(path: Path):
    mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    pyaldata = mat["pyaldata"]
    if isinstance(pyaldata, np.ndarray):
        return list(pyaldata.flat)
    return [pyaldata]


def normalize_spikes(spikes, unit_guide=None, ks_label=None):
    # this checks which way the spikes array is oriented, and is it correct in format, we want (time_bins, units) format
    s = np.asarray(spikes)
    if s.ndim != 2:
        raise ValueError(f"Expected 2D spikes, got {s.shape}") # just checks general shape

    n_units_meta = None
    if unit_guide is not None:
        ug = np.asarray(unit_guide)
        if ug.ndim >= 2:
            n_units_meta = ug.shape[0]
    if n_units_meta is None and ks_label is not None:
        kl = np.asarray(ks_label)
        if kl.ndim >= 1:
            n_units_meta = kl.shape[0]

    if n_units_meta is None:
        return s  # fallback if no metadata

    # already (time_bins, units)
    if s.shape[1] == n_units_meta:
        return s
    # transposed (units, time_bins) -> fix
    if s.shape[0] == n_units_meta:
        return s.T

    raise ValueError(
        f"Cannot align spikes shape {s.shape} with n_units_meta={n_units_meta}"
    )


def first_int_or_none(arr, idx):
    arr = np.asarray(arr)
    if arr.size == 0:
        return None
    flat = arr.reshape(-1)
    if idx >= flat.size:
        return None
    val = flat[idx]
    return int(val) if np.isfinite(val) else None


def insert_units(recording_key: dict, first_trial) -> None:
    rows = []
    for region, spikes_field, guide_field, label_field in REGION_FIELDS:
        spikes = as_2d_numeric(getattr(first_trial, spikes_field))
        n_units = spikes.shape[1]

        guides = np.asarray(getattr(first_trial, guide_field, np.array([])))
        labels = np.asarray(getattr(first_trial, label_field, np.array([])))

        for unit_idx in range(n_units):
            guide_row = guides[unit_idx] if guides.ndim >= 2 and unit_idx < guides.shape[0] else []
            label = labels[unit_idx] if labels.ndim >= 1 and unit_idx < labels.shape[0] else ""

            rows.append(
                {
                    **recording_key,
                    "brain_region": region,
                    "unit_idx": unit_idx,
                    "channel_best": first_int_or_none(guide_row, 0),
                    "unit_guide_1": first_int_or_none(guide_row, 0),
                    "unit_guide_2": first_int_or_none(guide_row, 1),
                    "ks_label": str(label),
                }
            )

    Unit.insert(rows, skip_duplicates=True)


def ingest_file(path: Path, max_trials: int | None = None, include_zero_counts: bool = False) -> None:
    path = path.expanduser().resolve()
    trials = load_trials(path)
    meta = parse_filename(path)
    recording_id = path.stem

    recording_row = {
        "recording_id": recording_id,
        "mat_file_path": str(path),
        "animal": meta["animal"],
        "session_label": meta["session_label"],
        "session_datetime": meta["session_datetime"],
        "part_index": meta["part_index"],
        "file_size_bytes": path.stat().st_size,
        "n_trials": len(trials),
    }
    Recording.insert1(recording_row, skip_duplicates=True) # put in 1 recording, per file

    recording_key = {"recording_id": recording_id}
    insert_units(recording_key, trials[0])

    total = len(trials) if max_trials is None else min(max_trials, len(trials))
    for trial_idx in range(total):
        trial = trials[trial_idx]

        # region 0 just to get n_time_bins consistently
        r0, sf0, gf0, lf0 = REGION_FIELDS[0]
        ref_spikes = normalize_spikes(
            getattr(trial, sf0),
            getattr(trial, gf0, None),
            getattr(trial, lf0, None),
        )

        trial_row = {
            **recording_key,
            "trial_idx": trial_idx,
            "trial_id": int(getattr(trial, "trial_id", trial_idx)),
            "trial_name": str(getattr(trial, "trial_name", "")),
            "trial_length": int(getattr(trial, "trial_length", ref_spikes.shape[0])),
            "bin_size_sec": float(getattr(trial, "bin_size", 0.0)),
            "n_time_bins": int(ref_spikes.shape[0]),
        }
        Trial.insert1(trial_row, skip_duplicates=True)

        count_rows = []
        for region, spikes_field, guide_field, label_field in REGION_FIELDS:
            spikes = normalize_spikes(
                getattr(trial, spikes_field),
                getattr(trial, guide_field, None),
                getattr(trial, label_field, None),
            )
            counts = np.asarray(spikes.sum(axis=0)).reshape(-1).astype(int)
            for unit_idx, n_spikes in enumerate(counts):
                if not include_zero_counts and n_spikes == 0:
                    continue
                count_rows.append(
                    {
                        **recording_key,
                        "trial_idx": trial_idx,
                        "brain_region": region,
                        "unit_idx": unit_idx,
                        "n_spikes": int(n_spikes),
                    }
                )

        if count_rows:
            TrialUnitSpikeCount.insert(count_rows, skip_duplicates=True)

        if (trial_idx + 1) % 25 == 0 or trial_idx == total - 1:
            print(f"[{path.name}] trial {trial_idx + 1}/{total}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest pyaldata .mat files into DataJoint (metadata + unit spike counts)."
    )
    parser.add_argument("mat_files", nargs="+", type=Path)
    parser.add_argument("--max-trials", type=int, default=None, help="Use first N trials for quick testing.") # for checking on small numbers of data
    parser.add_argument(
        "--include-zero-counts",
        action="store_true",
        help="Store rows where n_spikes == 0 (larger table).",
    )
    args = parser.parse_args()

    # example CL: python neuropixel_populate_data.py /Users/zosiasus/Documents/M046/M046_2024_12_19_13_30_pyaldata_0.mat --max-trials 3

    for mat_file in args.mat_files:
        print(f"Ingesting: {mat_file}")
        ingest_file(
            mat_file,
            max_trials=args.max_trials,
            include_zero_counts=args.include_zero_counts,
        )

    print("Done.")


if __name__ == "__main__":
    main()
