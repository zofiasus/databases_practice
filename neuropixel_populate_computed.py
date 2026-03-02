from neuropixel_data_schema import AllUnitMap, AllTrialSpikeCount

if __name__ == "__main__":
    AllUnitMap.populate()
    AllTrialSpikeCount.populate()
    print("Computed tables populated.")