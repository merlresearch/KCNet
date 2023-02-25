<!--
Copyright (C) 2023 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Kernel Correlation Network

## Features

This source code package contains the Python/Caffe implementation of our KCNet based on [Caffe](http://github.com/BVLC/caffe) for point cloud classification/segmentation training/testing/visualization.

## Installation

There are two folders in this package:

* modelnet

    ModelNet40/ModelNet10 object classification related codes and experiments.

* shapenet

    ShapeNet Part Segmentation related codes and experiments.

The codes were written and tested under **Ubuntu 14.04** with **CUDA 8.0**, **cuDNN 7.0**, **Python 2.7**, with the following packages:
- gdown>=3.4.6 ([pip install gdown](https://github.com/wkentaro/gdown))
- glog>=0.3.1 ([pip install glog](https://github.com/benley/python-glog))
- numpy>=1.11.0
- matplotlib>=1.5.1
- h5py==2.6.0
- Pillow>=1.1.7
- scipy>=0.17.0
- scikit_learn>=0.19.1
- scikit_image>=0.9.3
- pydot>=1.1.0

It also needs several tools for automatic download and compile dependencies:
- CMake>=3.5.1
- Git>=2.7.4

Setup
-----

1. Assume this project is located at:
```
    /homes/yourID/KCNet
```

2. Inside /homes/yourID/KCNet, execute the following
```
    python prepare_deps.py
```
which will download, compile, and install our modified Caffe (make sure you have setup your system following [the official Caffe install guide](http://caffe.berkeleyvision.org/installation.html)) in
```
    /homes/yourID/caffe
```
and download our Caffe front-end in
```
    /homes/yourID/caffecup
```

3. Inside /homes/yourID/KCNet, execute the following:
```
    python prepare_data.py
```
which will download ModelNet/ShapeNet data in
```
    /homes/yourID/KCNet/modelnet/data
    /homes/yourID/KCNet/shapenet/data
```
and our pre-trained networks and training logs in
```
    /homes/yourID/KCNet/modelnet/experiments
    /homes/yourID/KCNet/modelnet/log
    /homes/yourID/KCNet/shapenet/experiments
    /homes/yourID/KCNet/shapenet/log
```

Note: During the above step 2, if you encounter any:
1. **protobuf** related issue, make sure your install python protobuf with the version matched the output of `protoc --version`.

2. **Tkinter** related issue, make sure you install Tkinter by `sudo apt-get install python-tk`.
Then remove the automaticall downloaded caffe and rerun the step 2 command.

Notes on Network Short Names
----------------------------

|Short Name |Explanation|
|-----------|-----------|
|KConly     |Kernel Correlation (KC) with 16 responses, concat to X|
|GMonly     |Graph Max (GM) pooling without concatenation|
|GM2t4      |GM on X2 and concat to X4|
|GMconcat   |GM on X3 and concat to X4, plus GM on X4 and concat to P|
|KCt2/3/4   |KC concat to X2/3/4|
|KCtall     |KC concat to X2, X3,and X4|
|KCGM2t4    |KConly + GM2t4|
|KCt4GM2t4  |KCt4 + GM2t4|
|KC32       |KC only with 32 responses|
|KC32t3     |KC32 concat to X3|
|KC32GM2t4  |KC32 + GM2t4|

Also note that the **sigma** in this code is in fact $$2*\sigma^2$$ in our paper.


## Usage

Training Networks
-----------------

The codes were tested with an NVIDIA TitanX GPU with 12GB memory (although the max required GPU memory is at most 5GB in the default settings).

```
cd /homes/yourID/KCNet/modelnet
# Generate Caffe network files
python KC32GM2t4.py
# Start training (remove --no-srun if you use srun)
python KC32GM2t4.py brew --no-srun
# Clean intermediate files (only save the max-test-accuracy caffemodel)
python KC32GM2t4.py clean
```

Note that you can change **KC32GM2t4** into other network short names to train other networks.
Similarly, you can cd to **shapenet** folder to train shapenet part segmentation.

Visualization
-------------

```
# visualize hand-crafted kernels on ModelNet40
cd /homes/yourID/KCNet
python visualize.py

# visualize learned kernels on ShapeNet (remove --no-srun if you use srun)
cd /homes/yourID/KCNet/shapenet
python KCGM2t4.py vis --weight experiments/KCGM2t4/snapshot/KCGM2t4_Afl1e-3_iter_58625.caffemodel --no-srun
```

## Citation

If you use the software, please cite the following  ([TR2018-041](https://www.merl.com/publications/TR2018-041)):

```
@inproceedings{Shen2018jun,
author = {Shen, Yiru and Feng, Chen and Yang, Yaoqing and Tian, Dong},
title = {Mining Point Cloud Local Structures by Kernel Correlation and Graph Pooling},
booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
year = 2018,
month = jun,
url = {https://www.merl.com/publications/TR2018-041}
}
```

## Contact

Tim K Marks (<tmarks@merl.com>)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for our policy on contributions.

## License

Released under `AGPL-3.0-or-later` license, as found in the [LICENSE.md](LICENSE.md) file.

All files:

```
Copyright (c) 2018, 2023 Mitsubishi Electric Research Laboratories (MERL).

SPDX-License-Identifier: AGPL-3.0-or-later
```
