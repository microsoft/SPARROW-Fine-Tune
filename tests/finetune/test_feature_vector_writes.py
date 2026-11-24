from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import numpy as np

from sparrow_finetune import features


def test_overlapping_preparations_publish_complete_vectors(tmp_path, monkeypatch):
    rendezvous = Barrier(2)
    replace = features.os.replace

    def publish(source, destination):
        rendezvous.wait(timeout=5)
        replace(source, destination)

    monkeypatch.setattr(features.os, "replace", publish)
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(features._write_vector, tmp_path, "same.npy", np.ones(8)) for _ in range(2)]
        for job in jobs:
            job.result()
    np.testing.assert_array_equal(np.load(tmp_path / "same.npy"), np.ones(8))
    assert list(tmp_path.iterdir()) == [tmp_path / "same.npy"]
