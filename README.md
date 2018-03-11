# Interactive 3D Modeling with a Generative Adversarial Network

ArXiv: https://arxiv.org/abs/1706.05170. Published at International Conference on 3D Vision, 2017.

The code is adapted from 3dgan-release: https://github.com/zck119/3dgan-release. 

This repository consists of the training/generation code for our 3D-GAN-based framework on building an interactive modeling tool. The actual code for our web interface isn't here, but rather in https://github.com/jerryjliu/voxel-builder and https://github.com/jerryjliu/voxel-builder-server respectively. The training code is also not completely organized but I'll provide some pointers as to how to get started. 

The training code for the core GAN should be in main.py - take a look at the arguments to see how to train the model. An example training command would be: 
```
niter=1500 nskip=20 is32=0 checkpointn=0 glr=0.0025 dlr=0.00001 gpu=1 batchSize=100 data_name='full_dataset_voxels_64_chair' checkpointf='checkpoints_64chair100o' th main.lua
```

The training code for the projection network is in main_projection.py. An example training command would be: 
```
out_ext='feat18' feat_checkpointf='checkpoints_64chair100o_vaen2' feat_epochf='shapenet101_1500_net_D_split15' dropInput=0.5 genEpoch=1500 niter=2000 is32=0 gpu=4 data_name='full_dataset_voxels_64_chair' gen_checkpointf='checkpoints_64chair100o_vaen2' batchSize=50 plr=0.0001 th main_projection.lua
```

Note that the projection network needs to use a trained generator's output to measure the reconstruction loss between the input to the projection network and generated output. So you will need to specify `gen_checkpointf` as an argument to fetch a generator from the specified checkpoint. Moreover, the input/output are compared using L2 loss in some feature space (typically a classifer/discriminator), so you will also need to specify the `feat_epochf` argument. 

Once the two networks are trained, you can generate samples. I have written both Torch and PyTorch versions of `proj_generate.lua/py`, but the PyTorch version is the up-to-date one (don't use the Torch version). 
A generate command will look like this: 
```
python proj_generate.py --gpu 4 --input testvoxels --ckp checkpoints_64chair100o_vaen2 --ckgen 1500 --ckproj 1100 --ckext feat18 --optimize True --ckc checkpoints_64chair100o_vaen2 --ckclass 1500 --cksplit -1 --ckcext net_D
```
The optimize option is to optimize the projected sample further after feeding it once through the projection network. To replicate the results of the paper, you must specify discriminator activations (`--ckcext net_D`) as the target for optimization.  



