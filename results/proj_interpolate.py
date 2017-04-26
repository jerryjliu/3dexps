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
import random
import math
import graph_dist_similarity as gds

if __name__ == "__main__": 
    # parse args from command line
    parser = OptionParser()
    parser.add_option("--gpu",default=0,help="GPU id, starting from 1. Set it to 0 to run it in CPU mode.")
    parser.add_option('--input',default='test_chair1',help='the name of the input file')
    parser.add_option('--ckp',default='checkpoints_64chair_vaen2',help='checkpoint folder of gen model')
    parser.add_option('--ckgen', default='1450',help='checkpoint of the gen model')
    parser.add_option('--ckproj',default='229',help='checkpoint of the projection model')
    parser.add_option('--ckext',default='',help='extension to ckp to specify name of projection folder ( default is none )')
    parser.add_option('--gen_graph', default=False, help='whether to generate graph of similarity vs distance on the manifold. If true then need to set the below params.')
    parser.add_option('--ckc', default='checkpoints_64class100_5',help='checkpoint folder of classifier feature space')
    parser.add_option('--ckclass', default='200', help='checkpoint of split classifier')
    parser.add_option('--cksplit', default=9, help='split index of classifier')
    parser.add_option('--out',default='',help='specify full output file path  (if none put in local output/ folder)')
    parser.add_option('--outformat',default='mat',help='specify format of output (mat, pickle, json)')
    (opt,args) = parser.parse_args()
    
    (opt,args) = parser.parse_args()
    print(opt)
    if opt.gpu > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    data_dir = '/data/jjliu/models/proj_inputs_voxel/'

    tmpinput = sio.loadmat(os.path.join(data_dir, opt.input+'.mat'),variable_names=['off_volume'])
    tmpinput = torch.from_numpy(tmpinput['off_volume'])
    inp = torch.Tensor(1,1,64,64,64)
    inp.copy_(tmpinput)

    # load generator
    print('Loading network..')
    netG = pg.load_generator(opt.ckp, opt.ckgen, True)
    netP = pg.load_projection(opt.ckp, opt.ckgen, opt.ckproj, True, ext=opt.ckext)
    netC = None
    if opt.gen_graph:
        if opt.ckc != '' and opt.ckc is not None:
            netC = pg.load_split_classifier(opt.ckc, opt.ckclass, opt.cksplit, evaluate=True)

    if opt.gpu > 0:
        netG = netG.cuda()
        netP = netP.cuda()
        if netC is not None:
            netC = netC.cuda()
        inp = inp.cuda()

    # 1) project, and generate interpolations by varying various dimensions of the generated result
    npairs = 5
    ninter = 6
    nz = 200
    result, latent = pg.project_input(netP, netG, inp)
    latent.squeeze_(0)
    print(latent.size())
    print(result.__class__.__name__)
    if opt.gpu > 0:
        result_np = result.cpu().numpy()
        latent_np = latent.cpu().numpy()
    else:
        result_np = result.numpy()
        latent_np = latent.numpy()
    sio.savemat(opt.out + '_proj.mat', mdict={'inputs':latent_np, 'voxels': result_np})
    for i in range(0, npairs):
        randdim = int(math.floor(random.random() * nz))
        target = 2.5
        print(latent[randdim, 0, 0, 0])
        if latent[randdim, 0, 0, 0] > 0.0:
            target = -2.5
        olatent = latent.clone()
        olatent[randdim, 0, 0, 0] = target
        print(olatent.size())
        points, results = interp.interpolate_points(latent, olatent, ninter, netG, opt.gpu)
        if opt.gpu > 0:
            points_np = points.cpu().numpy()
            results_np = results.cpu().numpy()
        else:
            points_np = points.numpy()
            results_np = results.numpy()
        sio.savemat(opt.out + '_' + str(i) + '.mat', mdict={'inputs':points_np, 'voxels': results_np})

    #2) if generating graphs, generate graph of similarity vs distance from projected point on the manifold
    if opt.gen_graph:
        numsamples = 50
        dists, sims = gds.graph_distance_similarity(netC, netG, inp, latent, numsamples, opt.gpu)
        plt.figure()
        plt.scatter(dists, sims)
        plt.title('Similarity vs. Distance on the Manifold')
        plt.xlabel('L2 distance from projected latent vector z=P(x)') 
        plt.ylabel('Similarity to x')
        plt.show()
