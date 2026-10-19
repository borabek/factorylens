from pathlib import Path

import pytest

from factorylens.data import mvtec3d, subsets

STEMS = [f"{i:03d}" for i in range(60)]


def test_subsets_are_nested_and_sized():
    out = subsets.nested_subsets("bagel", STEMS)
    for per_k in out.values():
        prev: set[str] = set()
        for k in subsets.KS:
            cur = set(per_k[str(k)])
            assert len(cur) == k
            assert prev <= cur
            prev = cur


def test_subsets_do_not_depend_on_input_order():
    a = subsets.nested_subsets("bagel", STEMS)
    b = subsets.nested_subsets("bagel", list(reversed(STEMS)))
    assert a == b


def test_seeds_and_categories_differ():
    a = subsets.nested_subsets("bagel", STEMS)
    b = subsets.nested_subsets("tire", STEMS)
    assert a["0"]["10"] != a["1"]["10"]
    assert a["0"]["10"] != b["0"]["10"]


def test_too_few_samples():
    with pytest.raises(ValueError, match="only 3 training samples"):
        subsets.nested_subsets("bagel", STEMS[:3])


def test_build_reads_stems_from_disk(root):
    built = subsets.build(root, categories=["bagel"], ks=(1, 3))
    assert built["bagel"]["0"]["3"] == ["000", "001", "002"]
    with pytest.raises(ValueError):
        subsets.build(root, categories=["bagel"])  # the fake category has only 3 samples


def test_select_from_fake_dataset(root, tmp_path):
    train = mvtec3d.list_samples(root, "bagel", "train")
    built = subsets.build(root, categories=["bagel"], ks=(1, 2))
    path = tmp_path / "kshot.json"
    subsets.save(built, path)
    loaded = subsets.load(path)
    assert loaded == built

    one = subsets.select(train, loaded, k=1, seed=2)
    assert len(one) == 1 and one[0].rgb_path.stem in loaded["bagel"]["2"]["1"]
    assert subsets.select(train, loaded, k=None) == train
    assert subsets.select([], loaded, k=1) == []


def test_select_notices_missing_files(root):
    train = mvtec3d.list_samples(root, "bagel", "train")
    fake = {"bagel": {"0": {"2": ["000", "999"]}}}
    with pytest.raises(ValueError, match="expected 2 samples, found 1"):
        subsets.select(train, fake, k=2)


def test_committed_kshot_file_is_well_formed():
    """configs/kshot.json is what every k-shot result uses, so check it without the dataset."""
    committed = subsets.load(Path(__file__).parents[1] / "configs" / "kshot.json")
    assert set(committed) == set(mvtec3d.CATEGORIES)
    for category, per_seed in committed.items():
        assert set(per_seed) == {str(s) for s in subsets.SEEDS}
        n_train = mvtec3d.EXPECTED_COUNTS[category][0]
        for per_k in per_seed.values():
            prev: set[str] = set()
            for k in subsets.KS:
                cur = set(per_k[str(k)])
                assert len(cur) == k and prev <= cur
                assert all(0 <= int(stem) < n_train for stem in cur)
                prev = cur
