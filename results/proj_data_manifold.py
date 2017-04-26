import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.legacy.nn as lnn
import os
from optparse import OptionParser
import scipy.io as sio
from torch.utils.serialization import load_lua
import time
import proj_generate as pg
import numpy as np
from sklearn import decomposition
import matplotlib.pyplot as plt
import interpolate as interp

if __name__ == "__main__": 
    # parse args from command line
    parser = OptionParser()
    parser.add_option("--gpu",default=0,help="GPU id, starting from 1. Set it to 0 to run it in CPU mode.")
    parser.add_option('--ckp',default='checkpoints_64chair100o_vaen2',help='checkpoint folder of gen model')
    parser.add_option('--ckgen', default='1450',help='checkpoint of the gen model')
    parser.add_option('--ckproj',default='229',help='checkpoint of the projection model')
    parser.add_option('--ckext',default='',help='extension to ckp to specify name of projection folder ( default is none )')
    parser.add_option('--data_dir', default='full_dataset_voxels_64_chair', help='Directory of the data')
    (opt,args) = parser.parse_args()
    
    print(opt)
    if opt.gpu > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

    data_dir = '/data/jjliu/models'
    data_dir = os.path.join(data_dir, opt.data_dir)
    cache_dir = '/data/jjliu/cache'
    cache_file = os.path.join(cache_dir, 'proj_manifold_' + opt.data_dir + '_' + opt.ckproj + '.t7')

    inp = torch.Tensor(1,1,64,64,64)
    cats = os.listdir(data_dir)
    dim = 64
    nz = 200
    total = 0
    for i in range(len(cats)):
        cat = cats[i]
        cat_dir = os.path.join(data_dir, cat)
        cat_files = os.listdir(cat_dir)
        for j in range(len(cat_files)):
            total += 1
    latents = torch.Tensor(total, nz, 1, 1, 1)
    if not os.path.isfile(cache_file):
        # load generator
        print('Loading network..')
        netP = pg.load_projection(opt.ckp, opt.ckgen, opt.ckproj, True, ext=opt.ckext)

        if opt.gpu > 0:
            netP = netP.cuda()
            inp = inp.cuda()

        count = 0
        # go through all the data and project
        for i in range(len(cats)):
            cat = cats[i]
            cat_dir = os.path.join(data_dir, cat)
            cat_files = os.listdir(cat_dir)
            for j in range(len(cat_files)):
                print('count: ' + str(count) + '/' + str(total))
                full_mat_file = os.path.join(cat_dir, cat_files[j])
                tmpinput = sio.loadmat(full_mat_file,variable_names=['off_volume'])
                tmpinput = torch.from_numpy(tmpinput['off_volume'])
                inp.zero_()
                inp.copy_(tmpinput)
                latent = netP.forward(inp)
                latents[count] = latent
                count += 1
        torch.save(latents, cache_file)
    else:
        latents = torch.load(cache_file)
    latents_np = latents.numpy().reshape(total,nz)
    pca = decomposition.PCA(n_components=2)
    pca.fit(latents_np)
    latents_2d = pca.transform(latents_np)
    plt.figure()
    plt.scatter(latents_2d[:,0], latents_2d[:,1], s=1.0)
    plt.show()
