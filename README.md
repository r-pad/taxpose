# TAX-Pose: Task-Specific Cross-Pose Estimation for Robot Manipulation

## Installation.

Install a platform-specific version of torch.
```
pip install torch
```

Some experiments require pytorch geometric, which has a strange way of being installed.

```
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv torch_geometric -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
```

Install a platform-specific version of dgl.
```
pip install --pre dgl -f https://data.dgl.ai/wheels/cu117/repo.html
pip install --pre dglgo -f https://data.dgl.ai/wheels-test/repo.html
```
