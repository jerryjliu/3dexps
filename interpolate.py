import torch
import torch.nn as nn
import torch.legacy.nn
import os
from optparse import OptionParser
import scipy.io as sio
from torch.utils.serialization import load_lua
import time
import proj_generate as pg
import numpy as np
from sklearn import decomposition
import matplotlib.pyplot as plt

def interpolate_points(p1, p2, ninter, netG, outdir, outf):
    nz = p1.size()[0]
    dim = 64
    inputs = torch.Tensor(ninter, nz, 1, 1, 1)
    results = torch.Tensor(ninter, 1, dim, dim, dim)
    point = torch.zeros(nz, 1, 1, 1)
    inputs.zero_()
    results.zero_()
    for j in range(0, ninter):
        lam = j/(float(ninter) - 1)
        point.zero_()
        print(p1.size())
        point.add_(lam, p1)
        point.add_(1 - lam, p2)
        inputs[j].copy_(point)
    outputs = netG.forward(inputs)
    results.copy_(outputs)
    return inputs, results


def interpolate_randp(ninter, netG, outdir, outf):
    # for each pair, generate interpolation
    print('Setting points')
    print('Forward prop')
    point1 = torch.zeros(nz, 1, 1, 1)
    point2 = torch.zeros(nz, 1, 1, 1)
    point1.normal_(0,1)
    point2.normal_(0,1)
    inputs, results = interpolate_points(point1, point2, netG, outdir, outf)
    return inputs, results

def interpolate_ring(npairs, ninter, netG, outdir, outf, gen_graph=True):
    nz = 200
    points = torch.zeros(npairs, nz, 1, 1, 1)
    for i in range(0, npairs): 
        point = torch.zeros(nz, 1, 1, 1)
        point.normal_(0,1)
        points[i] = point
        #print(points[i].size())
    #print(points.size())
    center = points.mean(0)
    center.squeeze_(0)
    points_np = points.numpy().reshape(npairs, nz)
    center_np = center.numpy().reshape(nz)
    all_points_np = np.zeros((npairs, ninter, nz))
    for i in range(0, npairs):
        interpoints, results = interpolate_points(center, points[i], ninter, netG, outdir, outf)
        interpoints_np = interpoints.numpy().reshape(ninter, nz)
        all_points_np[i] = interpoints_np

    print(all_points_np.shape)
    if gen_graph:
        pca = decomposition.PCA(n_components=2)
        pca.fit(points_np)
        plt.figure(figsize=(8,6))
        for i in range(0, npairs):
            points_2d = pca.transform(all_points_np[i])
            plt.plot(points_2d[:,0], points_2d[:,1], label=('Interpolation '+str(i)))
        plt.legend(loc='upper left')
        plt.show()

if __name__ == "__main__": 
    # parse args from command line
    parser = OptionParser()
    parser.add_option("--gpu",default=0, help="GPU id, starting from 1. Set it to 0 to run it in CPU mode.")
    parser.add_option('--ckp',default='checkpoints_64chair_ref',help='checkpoint folder of gen model')
    parser.add_option('--ck', default='888',help='checkpoint of the gen model')
    parser.add_option('--npairs', default=7, type='int', help='number of pairs to interpolate between')
    parser.add_option('--ninter', default=10, type='int', help='number of interpolations between each pair')
    parser.add_option('--outdir', default='/data/jjliu/interpolate', help='output directory of interpolation')
    parser.add_option('--outf', default='', help='base name of output file')
    parser.add_option('--format', default='ring', help='format of interpolation (ring, rand: ring from center, or rand pairs)')
    
    (opt,args) = parser.parse_args()
    print(opt)
    assert(opt.ninter > 1)
    if opt.gpu > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    data_dir = '/data/jjliu/models/proj_inputs_voxel/'

    # load generator
    print('Loading network..')
    netG = pg.load_generator(opt.ckp, opt.ck, True)

    if opt.format == 'ring':
        interpolate_ring(opt.npairs, opt.ninter, netG, opt.outdir, opt.outf, gen_graph=True)
    else:
        for i in range(0, opt.npairs):
            inputs, results = interpolate_randp(opt.ninter, netG, opt.outdir, opt.outf)
            print(results.size())
            inputs_np = inputs.numpy()
            results_np = results.numpy()
            sio.savemat(os.path.join(opt.outdir, opt.outf + '_' + str(i) + '.mat'), mdict={'inputs':inputs_np, 'voxels': results_np})
            print('saving done')
