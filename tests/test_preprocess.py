from engagegraph.data.preprocess import Dataset, build_dataset, dataset_summary


def test_build_dataset_is_temporal_and_reproducible(parquet_dataset):
    dataset, config = parquet_dataset
    assert dataset.train.shape == dataset.test.shape
    assert dataset.n_users > 0 and dataset.n_items > 0
    assert dataset.test.nnz > 0
    train_pairs = set(zip(*dataset.train.nonzero(), strict=False))
    test_pairs = set(zip(*dataset.test.nonzero(), strict=False))
    assert train_pairs.isdisjoint(test_pairs)
    assert dataset.fingerprint
    assert "Users x items" in dataset_summary(dataset)

    rebuilt = build_dataset(config)
    assert rebuilt.user_ids == dataset.user_ids
    assert rebuilt.item_ids == dataset.item_ids
    assert rebuilt.fingerprint == dataset.fingerprint
    assert (rebuilt.train != dataset.train).nnz == 0


def test_dataset_save_load_roundtrip(parquet_dataset, tmp_path):
    dataset, _ = parquet_dataset
    dataset.save(tmp_path / "processed")
    loaded = Dataset.load(tmp_path / "processed")
    assert loaded.user_ids == dataset.user_ids
    assert loaded.item_ids == dataset.item_ids
    assert (loaded.train != dataset.train).nnz == 0
    assert (loaded.test != dataset.test).nnz == 0
    assert loaded.fingerprint == dataset.fingerprint


def test_no_test_item_was_seen_in_january(parquet_dataset):
    dataset, _ = parquet_dataset
    for user in dataset.eval_users:
        seen = set(dataset.train[user].indices)
        targets = set(dataset.test[user].indices)
        assert seen.isdisjoint(targets)
