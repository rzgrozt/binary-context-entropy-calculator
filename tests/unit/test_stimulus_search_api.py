import importlib


def test_public_api_when_package_is_imported_exposes_stimulus_search() -> None:
    # Given
    package = importlib.import_module("binary_entropy")

    # When
    exported = tuple(
        getattr(package, name, None)
        for name in (
            "StimulusSearchConfig",
            "StimulusConstraints",
            "SoftPreference",
            "SearchPartialReason",
            "search_stimuli",
            "create_complement_candidates",
            "match_stimuli",
            "assign_targets",
            "validate_stimuli",
            "stimulus_candidate_csv",
            "stimulus_scientific_csv",
            "stimulus_generator_config_json",
            "stimulus_experiment_csv",
        )
    )

    # Then
    assert all(item is not None for item in exported)
