# TAX-Pose: Task-Specific Cross-Pose Estimation for Robot Manipulation

## Installation.

Install a platform-specific version of torch.
```
pip install torch==1.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

Some experiments require pytorch geometric, which has a strange way of being installed.

```
pip install torch-scatter==2.0.9 torch-sparse==0.6.15 torch-cluster==1.6.0 torch-spline-conv==1.2.1 pyg_lib==0.1.0 -f https://data.pyg.org/whl/torch-1.11.0+cu113.html```

Install the pytorch3d binary.
```
pip install fvcore iopath
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu113_pyt1110/download.html

```

Install a platform-specific version of dgl.
```
pip install --pre dgl -f https://data.dgl.ai/wheels/cu117/repo.html
pip install --pre dglgo -f https://data.dgl.ai/wheels-test/repo.html
```
