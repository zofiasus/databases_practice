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
    mat_file_path: varchar(1024)
    ---
    animal: varchar(32)
    session_label: varchar(64)
    session_datetime: datetime
    part_index: tinyint 
    file_size_bytes: bigint 
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
