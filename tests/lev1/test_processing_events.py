"""Tests for the events processing module."""

import pandas as pd
import numpy as np

from network_glm.lev1.processing.events import (
    define_nuisance_trials,
    preprocess_events,
    stop_fail_violation,
)

class TestStopFailViolation:
    """Tests for stop_fail_violation function"""

    def test_missing_rt_produces_no_violation(self):
        """Tests that a missing rt for current (stop_fail) trial produces no 
violation"""
        df = pd.DataFrame(
            {
                "trial_id": ["test_trial", "test_trial"],
                "trial_type": ["go", "stop_failure"],
                "key_press": [1, np.nan],
                "correct_response": [1, np.nan],
                "response_time": [0.5, np.nan],
            }
        )
        output = stop_fail_violation(df)
        val = pd.to_numeric(output["stop_failure_violation"], errors="coerce")
        assert val.notna().sum() == 0

    def test_prev_not_go_produces_no_violation(self):
        """If the last test_trial is not a go, no violation should be 
produced"""
        df = pd.DataFrame(
            {
                "trial_id": ["test_trial", "test_trial"],
                "trial_type": ["stop_success", "stop_failure"],
                "key_press": [-1, 1],
                "correct_response": [-1, -1],
                "response_time": [-1, 0.6]
            }
        )
        output = stop_fail_violation(df)
        val = pd.to_numeric(output["stop_failure_violation"], errors="coerce")
        assert val.notna().sum() == 0

    def test_fixation_trials_are_skipped(self):
        """Intervening test_fixation rows should not effect pairing to 
prev_test_trial"""
        df = pd.DataFrame(
            {
                "trial_id": ["test_trial", "test_fixation", "test_trial"],
                "trial_type": ["go", "n/a", "stop_failure"],
                "key_press": [1, "n/a", 2],
                "correct_response": [1, "n/a", -1],
                "response_time": [0.4, np.nan, 0.6]
            }
        )
        output = stop_fail_violation(df)
        val = pd.to_numeric(output["stop_failure_violation"], errors="coerce")
        assert val.notna().sum() == 1

    def test_centering_and_sum_to_zero(self):
        """Multiple violations are mean-centered and sum to zero"""
        # raw amp: [0.3, 0.1, 0.5] -> mean 0.3 -> centered [0.0, -0.2, 0.2]
        df = pd.DataFrame(
            {
                "trial_id": ["test_trial", "test_trial", "test_trial", 
"test_trial", "test_trial", "test_trial"],
                "trial_type": ["go", "stop_failure", "go", "stop_failure", 
"go", "stop_failure"],
                "key_press": [1, 2, 2, 1, 2, 1],
                "correct_response": [1, -1, 2, -1, 2, -1],
                "response_time": [0.2, 0.5, 0.4, 0.5, 0.6, 1.1]
            }
        )
        output = stop_fail_violation(df)
        val = pd.to_numeric(output["stop_failure_violation"], errors="coerce").dropna().values
        assert len(val) == 3
        assert np.isclose(val.mean(), 0.0, atol=1e-12)
        assert np.isclose(val.sum(), 0.0, atol=1e-12)

    def go_rejected_for_rt_too_small(self):
        """Preceding go trial with rt < MIN_RT should be rejected"""
        df = pd.DataFrame({
            "trial_id": ["test_trial", "test_trial"],
            "trial_type": ["go", "stop_failure", "go", "stop_failure"],
            "key_press": [1, 2, 2, 2],
            "correct_response": [1, -1, 2, -1],
            "response_time": [0.18, 0.6, 0.19, 0.4],
        })
        output = stop_fail_violation(df)
        val = pd.to_numeric(output["stop_failure_violation"], errors="coerce")
        assert val.notna().sum() == 0

    def test_go_rejected_for_incorrect_accuracy(self):
        """A preceding go trial with key_press != correct_response should be rejected"""
        df = pd.DataFrame({
            "trial_id": ["test_trial", "test_trial"],
            "trial_type": ["go", "stop_failure", "go", "stop_failure"],
            "key_press": [1, 1, 1, 2],
            "correct_response": [2, -1, 2, -1],
            "response_time": [0.5, 0.8, 0.4, 0.5]
        })
        output = stop_fail_violation(df)
        assert pd.to_numeric(output["stop_failure_violation"], errors="coerce").notna().sum() == 0

    def test_zero_violation(self):
        """If there are no violations in a run, column should exist and remain all NaN"""
        df = pd.DataFrame({
            "trial_id": ["test_trial", "test_trial"],
            "trial_type": ["go", "go"],  #no stop_failure trials
            "key_press": [1, 1],
            "correct_response": [1, 1],
            "response_time": [0.4, 0.6],
        })
        output = stop_fail_violation(df)
        assert "stop_failure_violation" in output.columns
        val = pd.to_numeric(output["stop_failure_violation"], errors="coerce")
        assert val.notna().sum() == 0

    def test_single_violation(self):
        """Single violation makes it 0.0 after centering and so leaves column all NaN"""
        df = pd.DataFrame({
            "trial_id": ["test_trial", "test_trial"],
            "trial_type": ["go", "stop_failure"],
            "key_press": [1, 1],
            "correct_response": [1, -1],
            "response_time": [0.25, 0.3],
        })
        output = stop_fail_violation(df)
        val = pd.to_numeric(output["stop_failure_violation"], errors="coerce")
        assert val.notna().sum() == 0

    def test_two_consecutive_stop_failures(self):
        """Two consecutive stop_failure trials: the second stop_failure trial should stay NaN"""
        df = pd.DataFrame({
            "trial_id": ["test_trial", "test_trial", "test_trial"],
            "trial_type": ["go", "stop_failure", "stop_failure"],
            "key_press": [1, 1, 2],
            "correct_response": [1, -1, -1],
            "response_time": [0.3, 0.8, 0.4],
        })
        output = stop_fail_violation(df)
        val = pd.to_numeric(output["stop_failure_violation"], errors="coerce")
        #First stop_failure (index 1) should be defined due to previous valid go
        assert not np.isnan(val.iloc[1])
        #Second stop_failure (index 2) should be NaN
        assert np.isnan(val.iloc[2])



class TestPreprocessEvents:
    """Tests for preprocess_events function."""

    def test_preprocess_events_basic(self, sample_events_data):
        """Test basic event preprocessing."""
        processed = preprocess_events(sample_events_data, "cuedTS")

        # Check that junk column is added
        assert "junk" in processed.columns
        assert "na_trials" in processed.columns

        # Should not modify original dataframe
        assert processed is not sample_events_data

    def test_preprocess_events_negative_rt_handling(self):
        """Test handling of negative RT values."""
        events_data = pd.DataFrame(
            {
                "onset": [
                    20.0,
                    30.0,
                    40.0,
                    50.0,
                ],  # Use larger onsets to avoid negative after dummy adjustment
                "response_time": [0.5, -1.0, 0.7, -999],
                "trial_id": ["test_trial"] * 4,
                "trial_type": [
                    "tstay_cstay",
                    "tswitch_cswitch",
                    "tstay_cstay",
                    "tstay_cswitch",
                ],
            }
        )

        processed = preprocess_events(events_data, "cuedTS")

        # Should have 4 rows (no negative onsets after dummy adjustment)
        assert len(processed) == 4

        # Check by actual row position in processed dataframe
        # Row 1 (original index 1) has RT -1.0 -> should be junk=1, RT=NaN
        # Row 3 (original index 3) has RT -999 -> should be junk=1, RT=NaN
        junk_rows = processed[processed["junk"] == 1]
        non_junk_rows = processed[processed["junk"] == 0]

        assert len(junk_rows) == 2
        assert len(non_junk_rows) == 2

        # Check that negative RTs became NaN in junk rows
        assert all(pd.isna(junk_rows["response_time"]))

        # Check that positive RTs remain unchanged in non-junk rows
        assert 0.5 in non_junk_rows["response_time"].values
        assert 0.7 in non_junk_rows["response_time"].values


class TestDefineNuisanceTrials:
    """Tests for define_nuisance_trials function."""

    def test_define_nuisance_trials_task_switching(self):
        """Test nuisance trial definition for cuedTS."""
        events_data = pd.DataFrame(
            {
                "trial_id": ["test_trial"] * 6,
                "key_press": [1, 2, -1, 3, 1, 2],
                "correct_response": [1, 2, 1, 2, 1, 2],
                "response_time": [0.5, 0.15, -1, 0.6, 0.7, 0.8],
                "junk": [0, 0, 0, 0, 1, 0],
            }
        )

        nuisance_masks = define_nuisance_trials(events_data, "cuedTS")

        # Check expected nuisance types
        expected_keys = {"trial_filter", "bad_trials", "omission", "commission", "rt_too_fast"}
        assert set(nuisance_masks.keys()) == expected_keys

        # Verify specific trials
        assert nuisance_masks["bad_trials"].iloc[4]  # junk == 1
        assert nuisance_masks["omission"].iloc[2]  # key_press == -1
        assert nuisance_masks["commission"].iloc[3]  # wrong key press
        assert nuisance_masks["rt_too_fast"].iloc[1]  # RT < 0.2

    def test_define_nuisance_trials_stop_signal(self):
        """Test nuisance trial definition for stop-signal task."""
        events_data = pd.DataFrame(
            {
                "trial_type": ["go", "go", "stop_success", "go"],
                "key_press": [1, 2, -1, 3],
                "correct_response": [1, 2, -1, 2],
                "response_time": [0.5, 0.15, -1, 0.6],
                "junk": [0, 0, 0, 0],
            }
        )

        nuisance_masks = define_nuisance_trials(events_data, "stopSignal")

        # For stop-signal, trial_mask should be 'go' trials only
        assert nuisance_masks["omission"].sum() == 0  # No omissions in go trials
        assert nuisance_masks["commission"].iloc[3]  # Wrong response in go trial
        assert nuisance_masks["rt_too_fast"].iloc[1]  # Fast RT in go trial


def _stop_dual_df():
    return pd.DataFrame(
        {
            "trial_id": ["test_trial"] * 4,
            "trial_type": [
                "go_congruent",
                "go_incongruent",
                "stop_success_congruent",
                "stop_failure_incongruent",
            ],
            "key_press": [89, 71, -1, 71],
            "correct_response": [89, 71, -1, 89],
            "response_time": [0.5, 0.4, -1.0, 0.3],
        }
    )


def test_nonstop_dual_uses_test_trial_filter():
    df = pd.DataFrame(
        {
            "trial_id": ["test_cue", "test_trial", "test_trial"],
            "trial_type": ["n/a", "congruent_con", "incongruent_neg"],
            "key_press": [-1, 89, -1],
            "correct_response": [-1, 89, 71],
            "response_time": [-1.0, 0.5, 0.6],
        }
    )
    masks = define_nuisance_trials(df, "directedForgettingWFlanker")
    assert masks["omission"].tolist() == [False, False, True]


def test_stop_dual_go_restricted_omission():
    df = _stop_dual_df()
    masks = define_nuisance_trials(df, "stopSignalWFlanker")
    # successful stop (row 2, key_press == -1) must NOT be an omission (not a go trial)
    assert masks["omission"].tolist() == [False, False, False, False]
    assert masks["commission"].tolist() == [False, False, False, False]


def test_stop_dual_go_commission():
    df = _stop_dual_df()
    df.loc[0, "key_press"] = 71  # wrong on go_congruent -> commission
    masks = define_nuisance_trials(df, "stopSignalWDirectedForgetting")
    assert bool(masks["commission"].iloc[0]) is True
"""Tests for the events processing module."""

import pandas as pd

from network_glm.lev1.processing.events import (
    define_nuisance_trials,
    preprocess_events,
)


class TestPreprocessEvents:
    """Tests for preprocess_events function."""

    def test_preprocess_events_basic(self, sample_events_data):
        """Test basic event preprocessing."""
        processed = preprocess_events(sample_events_data, "cuedTS")

        # Check that junk column is added
        assert "junk" in processed.columns
        assert "na_trials" in processed.columns

        # Should not modify original dataframe
        assert processed is not sample_events_data

    def test_preprocess_events_negative_rt_handling(self):
        """Test handling of negative RT values."""
        events_data = pd.DataFrame(
            {
                "onset": [
                    20.0,
                    30.0,
                    40.0,
                    50.0,
                ],  # Use larger onsets to avoid negative after dummy 
adjustment
                "response_time": [0.5, -1.0, 0.7, -999],
                "trial_id": ["test_trial"] * 4,
                "trial_type": [
                    "tstay_cstay",
                    "tswitch_cswitch",
                    "tstay_cstay",
                    "tstay_cswitch",
                ],
            }
        )

        processed = preprocess_events(events_data, "cuedTS")

        # Should have 4 rows (no negative onsets after dummy adjustment)
        assert len(processed) == 4

        # Check by actual row position in processed dataframe
        # Row 1 (original index 1) has RT -1.0 -> should be junk=1, RT=NaN
        # Row 3 (original index 3) has RT -999 -> should be junk=1, RT=NaN
        junk_rows = processed[processed["junk"] == 1]
        non_junk_rows = processed[processed["junk"] == 0]

        assert len(junk_rows) == 2
        assert len(non_junk_rows) == 2

        # Check that negative RTs became NaN in junk rows
        assert all(pd.isna(junk_rows["response_time"]))

        # Check that positive RTs remain unchanged in non-junk rows
        assert 0.5 in non_junk_rows["response_time"].values
        assert 0.7 in non_junk_rows["response_time"].values


class TestDefineNuisanceTrials:
    """Tests for define_nuisance_trials function."""

    def test_define_nuisance_trials_task_switching(self):
        """Test nuisance trial definition for cuedTS."""
        events_data = pd.DataFrame(
            {
                "trial_id": ["test_trial"] * 6,
                "key_press": [1, 2, -1, 3, 1, 2],
                "correct_response": [1, 2, 1, 2, 1, 2],
                "response_time": [0.5, 0.15, -1, 0.6, 0.7, 0.8],
                "junk": [0, 0, 0, 0, 1, 0],
            }
        )

        nuisance_masks = define_nuisance_trials(events_data, "cuedTS")

        # Check expected nuisance types
        expected_keys = {"trial_filter", "bad_trials", "omission", 
"commission", "rt_too_fast"}
        assert set(nuisance_masks.keys()) == expected_keys

        # Verify specific trials
        assert nuisance_masks["bad_trials"].iloc[4]  # junk == 1
        assert nuisance_masks["omission"].iloc[2]  # key_press == -1
        assert nuisance_masks["commission"].iloc[3]  # wrong key press
        assert nuisance_masks["rt_too_fast"].iloc[1]  # RT < 0.2

    def test_define_nuisance_trials_stop_signal(self):
        """Test nuisance trial definition for stop-signal task."""
        events_data = pd.DataFrame(
            {
                "trial_type": ["go", "go", "stop_success", "go"],
                "key_press": [1, 2, -1, 3],
                "correct_response": [1, 2, -1, 2],
                "response_time": [0.5, 0.15, -1, 0.6],
                "junk": [0, 0, 0, 0],
            }
        )

        nuisance_masks = define_nuisance_trials(events_data, "stopSignal")

        # For stop-signal, trial_mask should be 'go' trials only
        assert nuisance_masks["omission"].sum() == 0  # No omissions in go 
trials
        assert nuisance_masks["commission"].iloc[3]  # Wrong response in go 
trial
        assert nuisance_masks["rt_too_fast"].iloc[1]  # Fast RT in go trial


def _stop_dual_df():
    return pd.DataFrame(
        {
            "trial_id": ["test_trial"] * 4,
            "trial_type": [
                "go_congruent",
                "go_incongruent",
                "stop_success_congruent",
                "stop_failure_incongruent",
            ],
            "key_press": [89, 71, -1, 71],
            "correct_response": [89, 71, -1, 89],
            "response_time": [0.5, 0.4, -1.0, 0.3],
        }
    )


def test_nonstop_dual_uses_test_trial_filter():
    df = pd.DataFrame(
        {
            "trial_id": ["test_cue", "test_trial", "test_trial"],
            "trial_type": ["n/a", "congruent_con", "incongruent_neg"],
            "key_press": [-1, 89, -1],
            "correct_response": [-1, 89, 71],
            "response_time": [-1.0, 0.5, 0.6],
        }
    )
    masks = define_nuisance_trials(df, "directedForgettingWFlanker")
    assert masks["omission"].tolist() == [False, False, True]


def test_stop_dual_go_restricted_omission():
    df = _stop_dual_df()
    masks = define_nuisance_trials(df, "stopSignalWFlanker")
    # successful stop (row 2, key_press == -1) must NOT be an omission (not a go trial)
    assert masks["omission"].tolist() == [False, False, False, False]
    assert masks["commission"].tolist() == [False, False, False, False]


def test_stop_dual_go_commission():
    df = _stop_dual_df()
    df.loc[0, "key_press"] = 71  # wrong on go_congruent -> commission
    masks = define_nuisance_trials(df, "stopSignalWDirectedForgetting")
    assert bool(masks["commission"].iloc[0]) is True
