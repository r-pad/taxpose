import os
import shutil
import tempfile

import numpy as np
import pytest
from rpad.pyg.dataset import CachedByKeyDataset

from taxpose.datasets.pm_placement import PlaceDataset


@pytest.mark.skipif(
    "PARTNET_MOBILITY_DATASET" not in os.environ,
    reason="PARTNET_MOBILITY_DATASET not set",
)
@pytest.mark.parametrize("mode", ["obs", "goal"])
def test_deterministic(mode):
    # Copy a few files into the temporary directory.
    root = os.environ["PARTNET_MOBILITY_DATASET"]

    def copy_into(dst):
        os.makedirs(os.path.join(dst, "raw"), exist_ok=False)
        shutil.copytree(
            os.path.join(root, "raw/12530/"),
            os.path.join(dst, "raw/12530/"),
            dirs_exist_ok=True,
        )

    scene_ids = [("12530", "bowl", "3", "top")]

    def make_dataset(tmp_root, seed):
        return CachedByKeyDataset(
            dset_cls=PlaceDataset,
            dset_kwargs=dict(
                root=tmp_root,
                scene_ids=scene_ids,
                randomize_camera=True,
                mode=mode,
                snap_to_surface=True,
                full_obj=True,
                even_downsample=True,
                rotate_anchor=True,
            ),
            data_keys=scene_ids,
            root=tmp_root,
            processed_dirname="test",
            n_repeat=2,
            n_workers=2,
            seed=seed,
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        copy_into(tmpdir)
        dset = make_dataset(tmp_root=tmpdir, seed=123456)
        data1 = dset[1]

    with tempfile.TemporaryDirectory() as tmpdir:
        copy_into(tmpdir)
        dset = make_dataset(tmp_root=tmpdir, seed=123456)
        data2 = dset[1]

    # Same seed -> same pos.
    assert np.array_equal(data1.anchor_pos, data2.anchor_pos)

    with tempfile.TemporaryDirectory() as tmpdir:
        copy_into(tmpdir)
        dset = make_dataset(tmp_root=tmpdir, seed=654321)
        data3 = dset[1]

    # Different seed -> different pos.
    assert not np.array_equal(data1.anchor_pos, data3.anchor_pos)
