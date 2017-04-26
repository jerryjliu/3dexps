import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.legacy.nn
from optparse import OptionParser
import scipy.io as sio
from torch.utils.serialization import load_lua
import time
import proj_generate as pg
import json
import numpy as np
import random

if __name__ == "__main__": 
    # parse args from command line
    parser = OptionParser()
    parser.add_option("--gpu",default=0,help="GPU id, starting from 1. Set it to 0 to run it in CPU mode.")
    parser.add_option("--data_dir",default='full_dataset_voxels_64_chair', help="data directory to load")
    parser.add_option('--nsamples',default=8, help="number of samples to project")
    parser.add_option('--ckp',default='checkpoints_64chair_ref',help='checkpoint folder of gen model')
    parser.add_option('--ckgen', default='888',help='checkpoint of the gen model')
    parser.add_option('--ckproj',default='40',help='checkpoint of the projection model')
    parser.add_option('--ckext',default='',help='extension to ckp to specify name of projection folder ( default is none )')
    parser.add_option('--outdir', default='/home/jjliu/Documents/3dexps/output', help='output directory of input and projected samples')
    (opt,args) = parser.parse_args()
    print(opt)
    if opt.gpu > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    data_dir = '/data/jjliu/models/'
    data_dir = os.path.join(data_dir, opt.data_dir)
    assert(os.path.isdir(data_dir))
    cats = os.listdir(data_dir)

    # load generator / projection
    print('Loading network..')
    netG = pg.load_generator(opt.ckp, opt.ckgen, True)
    netP = pg.load_projection(opt.ckp, opt.ckgen, opt.ckproj, True, ext=opt.ckext)

    # load from samples
    opt.nsamples = int(opt.nsamples)
    results = torch.Tensor(opt.nsamples, 1, 64, 64, 64)
    inps = torch.Tensor(opt.nsamples,1,64,64,64)
    latents = torch.Tensor(opt.nsamples,200,1,1,1)
    if opt.gpu > 0:
      netG = netG.cuda()
      netP = netP.cuda()
      inps = inps.cuda()

    inps.zero_()
    for i in range(opt.nsamples):
        randi = int(random.random() * len(cats))
        cat = cats[randi]
        cat_dir = os.path.join(data_dir, cat)
        cat_files = os.listdir(cat_dir)

        randi2 = int(random.random() * len(cat_files))
        cat_file = cat_files[randi2]
        full_cat_file = os.path.join(cat_dir, cat_file)
        tmpinput = sio.loadmat(full_cat_file, variable_names=['off_volume'])
        tmpinput = torch.from_numpy(tmpinput['off_volume'])
        inps[i].copy_(tmpinput)

    for i in range(opt.nsamples):
        output, latent = pg.project_input(netP, netG, inps[i].unsqueeze_(0))
        results[i] = output
    
    inps_np = inps.cpu().numpy()
    results_np = results.cpu().numpy()
    latents_np = latents.cpu().numpy()
    cur_time = int(time.time())
    cur_times = '' + str(cur_time)
    rfname = 'proj_' + cur_times + '_real'
    gfname = 'proj_' + cur_times + '_gen'
    sio.savemat(os.path.join(opt.outdir, rfname+'.mat'), mdict={'inputs':latents_np, 'voxels': inps_np})
    sio.savemat(os.path.join(opt.outdir, gfname+'.mat'), mdict={'inputs':latents_np, 'voxels': results_np})
