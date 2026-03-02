from neuropixel_data_schema import Recording, Trial, Unit, TrialUnitSpikeCount


def main():
    print("=== Row Counts ===")
    print("Recording:", len(Recording()))
    print("Trial:", len(Trial()))
    print("Unit:", len(Unit()))
    print("TrialUnitSpikeCount:", len(TrialUnitSpikeCount()))

    print("\n=== Orphan Checks (should be 0) ===")
    print("Trial - Recording:", len(Trial - Recording))
    print("Unit - Recording:", len(Unit - Recording))
    print("SpikeCount - Trial:", len(TrialUnitSpikeCount - Trial))
    print("SpikeCount - Unit:", len(TrialUnitSpikeCount - Unit))

    print("\n=== Basic Value Checks ===")
    print("Negative n_spikes:", len(TrialUnitSpikeCount & "n_spikes < 0"))
    print("Zero n_time_bins:", len(Trial & "n_time_bins <= 0"))
    print("Zero bin_size_sec:", len(Trial & "bin_size_sec <= 0"))

    print("\n=== Per-Recording Summary ===")
    print(Recording.aggr(Trial, n_trials_loaded="count(*)").to_pandas())

    print("\n=== Per-Region Unit Counts ===")
    u = Unit.to_pandas().reset_index() # unit table
    print(u.groupby(["recording_id", "brain_region"]).size().rename("n_units").reset_index())

'''
THIS PRINTED :) 
Connected: True
Schema ready: neuropixel_data_pipeline
=== Row Counts ===
Recording: 2
Trial: 796
Unit: 2274
TrialUnitSpikeCount: 530774

=== Orphan Checks (should be 0) ===
Trial - Recording: 0
Unit - Recording: 0
SpikeCount - Trial: 0
SpikeCount - Unit: 0

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
'''

if __name__ == "__main__":
    main()