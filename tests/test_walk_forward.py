"""
test_walk_forward.py
====================
Unit tests for src/evaluation/walk_forward.py

Checks:
- Temporal ordering of splits (no test-before-train)
- Minimum training size enforcement
- Expanding vs. rolling window modes
"""

import pytest
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, "..")

from src.evaluation.walk_forward import WalkForwardValidator


def make_index(n: int = 500) -> pd.DatetimeIndex:
    return pd.date_range("2015-01-01", periods=n, freq="B", tz="UTC")


class TestWalkForwardSplits:

    def test_train_always_precedes_test(self):
        idx = make_index()
        wfv = WalkForwardValidator(n_splits=5, window_type="expanding", min_train_size=50)
        for train_idx, test_idx in wfv.generate_splits(idx):
            assert train_idx[-1] < test_idx[0], (
                "Training end must come before test start."
            )

    def test_no_overlap_between_train_and_test(self):
        idx = make_index()
        wfv = WalkForwardValidator(n_splits=5, window_type="expanding", min_train_size=50)
        for train_idx, test_idx in wfv.generate_splits(idx):
            overlap = set(train_idx) & set(test_idx)
            assert len(overlap) == 0, "Train and test indices must not overlap."

    def test_number_of_folds(self):
        idx = make_index(n=600)
        wfv = WalkForwardValidator(n_splits=5, window_type="expanding", min_train_size=50)
        splits = list(wfv.generate_splits(idx))
        assert len(splits) == 5

    def test_expanding_window_grows(self):
        idx = make_index(n=600)
        wfv = WalkForwardValidator(n_splits=5, window_type="expanding", min_train_size=50)
        train_sizes = [len(t) for t, _ in wfv.generate_splits(idx)]
        assert all(
            train_sizes[i] <= train_sizes[i + 1]
            for i in range(len(train_sizes) - 1)
        ), "Expanding window training size must be non-decreasing."

    def test_minimum_train_size_enforced(self):
        idx = make_index(n=100)
        min_size = 80  # Too large for most folds
        wfv = WalkForwardValidator(n_splits=5, window_type="expanding", min_train_size=min_size)
        splits = list(wfv.generate_splits(idx))
        for train_idx, _ in splits:
            assert len(train_idx) >= min_size

    def test_rolling_window_fixed_size(self):
        idx = make_index(n=600)
        wfv = WalkForwardValidator(
            n_splits=5,
            window_type="rolling",
            train_window=100,
            min_train_size=50,
        )
        train_sizes = [len(t) for t, _ in wfv.generate_splits(idx)]
        for sz in train_sizes:
            assert sz == 100, "Rolling window must have fixed training size."

    def test_rolling_requires_train_window(self):
        with pytest.raises(ValueError, match="train_window"):
            WalkForwardValidator(n_splits=5, window_type="rolling")

    def test_gap_is_respected(self):
        idx = make_index(n=400)
        gap = 5
        wfv = WalkForwardValidator(n_splits=4, window_type="expanding",
                                   gap=gap, min_train_size=30)
        for train_idx, test_idx in wfv.generate_splits(idx):
            actual_gap = test_idx[0] - train_idx[-1] - 1
            assert actual_gap >= gap, (
                f"Gap of {gap} periods not respected: actual gap = {actual_gap}"
            )
