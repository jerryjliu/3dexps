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
    parser.add_option('--ckc', default='checkpoints_64class100_5',help='checkpoint folder of classifier feature space')
    parser.add_option('--ckclass', default='200', help='checkpoint of split classifier')
    parser.add_option('--cksplit', default=9, help='split index of classifier')
    parser.add_option('--data_dir', default='full_dataset_voxels_64_chair', help='Directory of the data')
    (opt,args) = parser.parse_args()
    
    print(opt)
    if opt.gpu > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

    data_dir = '/data/jjliu/models'
    data_dir = os.path.join(data_dir, opt.data_dir)
    cache_dir = '/data/jjliu/cache'
    cache_file = os.path.join(cache_dir, 'proj_feat_' + opt.data_dir + '_' + opt.ckclass + '_' + opt.cksplit + '.t7')

    inp = torch.Tensor(1,1,64,64,64)
    cats = os.listdir(data_dir)
    dim = 64
    total = 0
    for i in range(len(cats)):
        cat = cats[i]
        cat_dir = os.path.join(data_dir, cat)
        cat_files = os.listdir(cat_dir)
        for j in range(len(cat_files)):
            total += 1
    feats = None
    if not os.path.isfile(cache_file):
        # load generator
        print('Loading network..')
        netC = pg.load_split_classifier(opt.ckc, opt.ckclass, opt.cksplit, evaluate=True)
        if opt.gpu > 0:
            netC = netC.cuda()
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
                outfeat = netC.forward(inp)
                print(outfeat.size())
                if feats is None:
                    sizeArr = outfeat.size()
                    feats = torch.Tensor(total, sizeArr[1], sizeArr[2], sizeArr[3], sizeArr[4])
                feats[count] = outfeat
                count += 1
        torch.save(feats, cache_file)
    else:
        feats = torch.load(cache_file)
    feats_np = feats.numpy().reshape(total,-1)
    pca = decomposition.PCA(n_components=2)
    pca.fit(feats_np)
    feats_2d = pca.transform(feats_np)
    plt.figure()
    plt.scatter(feats_2d[:,0], feats_2d[:,1], s=1.0)
    plt.show()
