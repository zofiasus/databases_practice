import datajoint as dj

# Connect first so config/credentials errors fail immediately.
conn = dj.conn()
print("Connected:", conn.is_connected)

# Dedicated schema for Neuropixels practice data.
schema = dj.Schema("neuropixel_data_pipeline")
print("Schema ready:", schema.database)


@schema
class Recording(dj.Manual):
    definition = """
    # One row per source .mat file
    recording_id: varchar(128)
    ---
    mat_file_path: varchar(1024)
    animal: varchar(32)
    session_label: varchar(64)
    session_datetime: datetime
    part_index: int
    file_size_bytes: int
    n_trials: int
    """


@schema
class Trial(dj.Manual):
    definition = """
    # Trial metadata for each row in pyaldata
    -> Recording
    trial_idx: int unsigned
    ---
    trial_id=null: int
    trial_name='': varchar(128)
    trial_length=null: int
    bin_size_sec: float
    n_time_bins: int unsigned
    """


@schema
class Unit(dj.Manual):
    definition = """
    # Unit identity per brain region for a recording
    -> Recording
    brain_region: varchar(24)
    unit_idx: int unsigned
    ---
    channel_best=null: int
    unit_guide_1=null: int
    unit_guide_2=null: int
    ks_label='': varchar(32)
    """


@schema
class TrialUnitSpikeCount(dj.Manual):
    definition = """
    # Per-trial, per-unit total spikes (compact summary from binned spike matrix)
    -> Trial
    -> Unit
    ---
    n_spikes: int unsigned
    """

@schema
class AllUnitMap(dj.Computed):
    definition = """
    # Concatenated unit index for synthetic 'all' region (the raw data 'all' spike data is incorrect)
    -> Recording
    all_unit_idx: int
    ---
    source_region: varchar(24)
    source_unit_idx: int
    """

    key_source = Recording

    def make(self, key): # DataJoint calls this once per key in key_source (here: each recording)
        base = (
            Unit & key # restricts Unit table to that one recording
            & 'brain_region in ("Thal","CP","MOp","SSp_ll")' # keeps only base regions (excludes synthetic all)
        ).fetch("brain_region", "unit_idx", order_by="brain_region, unit_idx") # gets two aligned arrays, sorted
        rows = [] # make rows to insert to AllUnitMap
        for i, (region, unit_idx) in enumerate(zip(*base)):
            rows.append({
                **key,
                "all_unit_idx": i,
                "source_region": region,
                "source_unit_idx": int(unit_idx),
            })
        self.insert(rows)

        # print progress
        print(f"AllUnitMap inserted {len(rows)} rows for {key['recording_id']}")

                
        
        

@schema
class AllTrialSpikeCount(dj.Computed): # this will extend the TrialUnitSpikeCount by the 'all' spikes data
    definition = """
    -> Trial
    -> AllUnitMap
    ---
    n_spikes: int
    """

    key_source = Trial * AllUnitMap

    def make(self, key):# DataJoint calls this once per row in key_source (Trial * AllUnitMap)
        # Get source mapping for this synthetic all-unit
        source_region, source_unit_idx = (AllUnitMap & key).fetch1(
            "source_region", "source_unit_idx"
        )
        # Restrict TrialUnitSpikeCount only by attributes it actually has
        trial_key = {k: key[k] for k in Trial.primary_key}
        # Find the matching spike-count row in raw/unit-level data for that trial + mapped source unit, then read its n_spikes
        rel = (
            TrialUnitSpikeCount
            & trial_key
            & {"brain_region": source_region, "unit_idx": int(source_unit_idx)}
        )

        vals = rel.fetch("n_spikes")
        n = int(vals[0]) if len(vals) else 0 # account for empty rows

        self.insert1({**key, "n_spikes": n})

        # print every 500 trial indices for quick progress
        if key["trial_idx"] % 500 == 0 and key["all_unit_idx"] == 0:
            print(f"AllTrialSpikeCount progress: recording={key['recording_id']} trial_idx={key['trial_idx']}")


