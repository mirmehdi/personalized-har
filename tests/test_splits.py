# tests/test_splits.py
# Guards the one property the whole project depends on:
# no person may appear in both train and test of the same fold.
import pytest
import numpy as np                          # arrays + set operations on indices
from har.data.splits import subject_folds   # the splitter under test


def check_no_leakage(folds, subject_ids):
    """Reusable check: raises AssertionError if any fold mixes a person across the split."""
    for train_idx, test_idx in folds:                       # walk every fold
        train_people = set(subject_ids[train_idx])          # who is in this fold's training set
        test_people = set(subject_ids[test_idx])            # who is in this fold's test set
        overlap = train_people & test_people                # anyone on both sides is a leak
        assert not overlap, f"leaked subjects: {overlap}"   # fail loudly, naming the culprits


def test_no_subject_leakage():
    """The real splitter must never leak a person across a fold."""
    ids = np.repeat([0, 1, 2, 3], 5)                        # 4 people, 5 windows each
    check_no_leakage(subject_folds(ids), ids)               # reuse the check above


def test_every_subject_held_out_once():
    """Every person must be the test subject in exactly one fold."""
    ids = np.repeat([0, 1, 2, 3], 5)                        # same dummy population
    held_out = [set(ids[te]) for _, te in subject_folds(ids)]   # the test person per fold
    assert all(len(h) == 1 for h in held_out)               # exactly one person per fold
    union = set().union(*held_out)                          # everyone who was ever held out
    assert union == set(ids)                                # must cover the whole population
    assert len(held_out) == len(set(ids))                   # and one fold per person, no repeats


def test_indices_are_conserved():
    """Train + test together must cover every window exactly once — nothing lost or duplicated."""
    ids = np.repeat([0, 1, 2, 3], 5)                        # 20 windows total
    for train_idx, test_idx in subject_folds(ids):          # check fold by fold
        combined = np.concatenate([train_idx, test_idx])    # everything the fold uses
        assert len(combined) == len(ids)                    # no window dropped or double-counted
        assert set(combined) == set(range(len(ids)))        # and the exact index set is intact

def leaky_split(subject_ids, n_folds=4):
    """A deliberately WRONG splitter: shuffles windows and ignores who they came from,
    so one person's windows land on both sides. This is the mistake we're guarding against."""
    rng = np.random.default_rng(0)                       # fixed seed = same shuffle every run
    shuffled = rng.permutation(len(subject_ids))         # window positions in random order
    for fold in np.array_split(shuffled, n_folds):       # chop the shuffle into n test blocks
        test_idx = fold                                  # this block is "test"
        train_idx = np.setdiff1d(shuffled, test_idx)     # everything else is "train"
        yield train_idx, test_idx                        # note: subject boundaries ignored


def test_check_catches_leakage():
    """The guard must actually bite: a random, non-grouped split MUST be rejected."""
    ids = np.repeat([0, 1, 2, 3], 5)                     # same dummy population as before
    with pytest.raises(AssertionError):                  # we EXPECT the check to fail here
        check_no_leakage(leaky_split(ids), ids)          # if it passes, the guard is useless