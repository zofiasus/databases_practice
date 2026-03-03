from neuropixel_data_schema import Recording, Trial, Unit, TrialUnitSpikeCount


def main():
    recording_rel = Recording()
    trial_rel = Trial()
    unit_rel = Unit()
    spike_rel = TrialUnitSpikeCount()

    print("=== Row Counts ===")
    print("Recording:", len(recording_rel))
    print("Trial:", len(trial_rel))
    print("Unit:", len(unit_rel))
    print("TrialUnitSpikeCount:", len(spike_rel))

    print("\n=== Orphan Checks (should be 0) ===")
    print("Trial without Recording:", len(Trial()) - len(Trial() * Recording()))
    print("Unit without Recording:", len(Unit()) - len(Unit() * Recording()))
    print("SpikeCount without Trial:", len(TrialUnitSpikeCount()) - len(TrialUnitSpikeCount() * Trial()))
    print("SpikeCount without Unit:", len(TrialUnitSpikeCount()) - len(TrialUnitSpikeCount() * Unit()))

    print("\n=== Basic Value Checks ===")
    print("Negative n_spikes:", len(spike_rel & "n_spikes < 0"))
    print("Zero n_time_bins:", len(trial_rel & "n_time_bins <= 0"))
    print("Zero bin_size_sec:", len(trial_rel & "bin_size_sec <= 0"))

    print("\n=== Per-Recording Summary ===")
    print(Recording.aggr(Trial, n_trials_loaded="count(*)").to_pandas())

    print("\n=== Per-Region Unit Counts ===")
    u = Unit.to_pandas().reset_index()
    print(u.groupby(["recording_id", "brain_region"]).size().rename("n_units").reset_index())

    print("\n=== Trial Type Counts ===")
    t = Trial.to_pandas().reset_index()
    print(
        t.groupby(["recording_id", "trial_type"], dropna=False)
        .size()
        .rename("n_trials")
        .reset_index()
        .sort_values(["recording_id", "n_trials"], ascending=[True, False])
    )

    print("\n=== Solenoid Fields Missingness by Trial Type ===")
    sol_missing = (
        t.assign(
            sol_direction_missing=t["sol_direction"].isna(),
            idx_sol_direction_missing=t["idx_sol_direction"].isna(),
            idx_sol_on_missing=t["idx_sol_on"].isna(),
        )
        .groupby(["recording_id", "trial_type"], dropna=False)[
            ["sol_direction_missing", "idx_sol_direction_missing", "idx_sol_on_missing"]
        ]
        .mean()
        .reset_index()
    )
    print(sol_missing)


if __name__ == "__main__":
    main()

'''
THIS PRINTED :) 

=== Row Counts ===
Recording: 2
Trial: 796
Unit: 2274
TrialUnitSpikeCount: 905052

=== Orphan Checks (should be 0) ===
Trial without Recording: 0
Unit without Recording: 0
SpikeCount without Trial: 0
SpikeCount without Unit: 0

=== Basic Value Checks ===
Negative n_spikes: 0
Zero n_time_bins: 0
Zero bin_size_sec: 0

=== Per-Recording Summary ===
                                  n_trials_loaded
recording_id                                     
M046_2024_12_19_13_30_pyaldata_0              398
M046_2024_12_19_13_30_pyaldata_1              398

=== Per-Region Unit Counts ===
                       recording_id brain_region  n_units
0  M046_2024_12_19_13_30_pyaldata_0           CP      455
1  M046_2024_12_19_13_30_pyaldata_0          MOp      281
2  M046_2024_12_19_13_30_pyaldata_0       SSp_ll      198
3  M046_2024_12_19_13_30_pyaldata_0         Thal      184
4  M046_2024_12_19_13_30_pyaldata_0          all       19
5  M046_2024_12_19_13_30_pyaldata_1           CP      455
6  M046_2024_12_19_13_30_pyaldata_1          MOp      281
7  M046_2024_12_19_13_30_pyaldata_1       SSp_ll      198
8  M046_2024_12_19_13_30_pyaldata_1         Thal      184
9  M046_2024_12_19_13_30_pyaldata_1          all       19

=== Trial Type Counts ===
                       recording_id  trial_type  n_trials
1  M046_2024_12_19_13_30_pyaldata_0  intertrial       199
2  M046_2024_12_19_13_30_pyaldata_0       trial       198
0  M046_2024_12_19_13_30_pyaldata_0        free         1
5  M046_2024_12_19_13_30_pyaldata_1       trial       199
4  M046_2024_12_19_13_30_pyaldata_1  intertrial       198
3  M046_2024_12_19_13_30_pyaldata_1        free         1

=== Solenoid Fields Missingness by Trial Type ===
                       recording_id  trial_type  sol_direction_missing  idx_sol_direction_missing  idx_sol_on_missing
0  M046_2024_12_19_13_30_pyaldata_0        free                    1.0                        1.0                 1.0
1  M046_2024_12_19_13_30_pyaldata_0  intertrial                    0.0                        0.0                 1.0
2  M046_2024_12_19_13_30_pyaldata_0       trial                    0.0                        0.0                 0.0
3  M046_2024_12_19_13_30_pyaldata_1        free                    1.0                        1.0                 1.0
4  M046_2024_12_19_13_30_pyaldata_1  intertrial                    0.0                        0.0                 1.0
5  M046_2024_12_19_13_30_pyaldata_1       trial                    0.0                        0.0                 0.0

'''
