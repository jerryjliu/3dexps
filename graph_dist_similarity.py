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

def graph_distance_similarity(netC, netG, inpobj, inplatent, numsamples, gpu=0):
        simCriterion = lnn.MSECriterion()
        dists = np.zeros(numsamples)
        sims = np.zeros(numsamples)
        if gpu > 0:
            simCriterion = simCriterion.cuda()
            latent_np = inplatent.cpu().numpy()
        else:
            latent_np = inplatent.numpy()
        latent_np = latent_np.reshape(inplatent.size(0))
        for i in range(0, numsamples): 
            noise = torch.rand(inplatent.size(0), 1, 1, 1)
            noise = noise.mul((i+1) * 0.05)
            if gpu > 0:
                noise = noise.cuda()
            olatent = inplatent.add(noise)
            olatent.unsqueeze_(0)
            if gpu > 0:
                olatent_np = olatent.cpu().numpy().reshape(inplatent.size(0))
            else:
                olatent_np = olatent.numpy().reshape(inplatent.size(0))
            oresult = netG.forward(olatent)
            difflatent_np = olatent_np - latent_np
            dist = np.linalg.norm(difflatent_np)
            feat1 = netC.forward(inpobj)
            feat1 = feat1.clone()
            feat2 = netC.forward(oresult)
            similarity = simCriterion.forward(feat2, feat1)
            dists[i] = dist
            sims[i] = similarity
            print(str(dist) + " " + str(similarity))
        return dists, sims

if __name__ == "__main__": 
    # parse args from command line
    parser = OptionParser()
    parser.add_option("--gpu",default=0,help="GPU id, starting from 1. Set it to 0 to run it in CPU mode.")
    parser.add_option('--ckp',default='checkpoints_64chair100o_vaen2',help='checkpoint folder of gen model')
    parser.add_option('--ckgen', default='1450',help='checkpoint of the gen model')
    parser.add_option('--ckc', default='checkpoints_64class100_5',help='checkpoint folder of classifier feature space')
    parser.add_option('--ckclass', default='200', help='checkpoint of split classifier')
    parser.add_option('--cksplit', default=9, help='split index of classifier')
    parser.add_option('--numsamples', default=50, help='numsamples around a given point')
    (opt,args) = parser.parse_args()
    
    (opt,args) = parser.parse_args()
    print(opt)
    if opt.gpu > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

    # initialize variables
    print('Loading network..')
    netG = pg.load_generator(opt.ckp, opt.ckgen, True)
    netC = pg.load_split_classifier(opt.ckc, opt.ckclass, opt.cksplit, evaluate=True)
    nz = 200
    center = torch.Tensor(1, nz, 1, 1, 1)
    center.normal_(0,1)

    if opt.gpu > 0:
        netG = netG.cuda()
        netC = netC.cuda()
        center = center.cuda()

    # plot graph
    inpobj = netG.forward(center)
    inpobj = inpobj.clone()
    dists, sims = graph_distance_similarity(netC, netG, inpobj, center.squeeze(0), int(opt.numsamples), opt.gpu)
    plt.figure()
    plt.scatter(dists, sims)
    plt.title('Similarity vs. Distance on the Manifold')
    plt.xlabel('L2 distance from random latent vector z') 
    plt.ylabel('Similarity to G(z)')
    plt.show()
