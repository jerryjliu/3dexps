import torch
import torch.nn as nn
import torch.legacy.nn
import torch.legacy.optim
import os
from optparse import OptionParser
import scipy.io as sio
from torch.utils.serialization import load_lua
import time
import json
import random
import numpy as np
import proj_generate as pg

if __name__ == "__main__": 
    # parse args from command line
    parser = OptionParser()
    parser.add_option("--gpu",default=0,help="GPU id, starting from 1. Set it to 0 to run it in CPU mode.")
    parser.add_option('--input',default='/data/jjliu/models/proj_outputs_voxel/testvoxels_out',help='the name of the input file (.mat format with "input" and "voxels"')
    parser.add_option('--informat',default='mat',help='format of input object (mat, ndarray)')
    parser.add_option('--sample_size',default=-1, help='how many samples from the training data (default is all)')
    parser.add_option('--nresults', default=3,help='how many results to retrieve')
    parser.add_option('--data_dir', default='full_dataset_voxels_64', help='name of the data folder from which to retrieve your samples (.mat format with "off_volume")')
    parser.add_option('--ckc', default='checkpoints_64class100_5',help='checkpoint folder of classifier feature space')
    parser.add_option('--ckclass', default='200', help='checkpoint of split classifier')
    parser.add_option('--cksplit', default=9, help='split index of classifier')
    parser.add_option('--ckcext', default='net_C', help='extension of the classifier model (typically net_C but can be arbitrary)')
    (opt,args) = parser.parse_args()
    print(opt)
    if opt.gpu > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    data_dir = '/data/jjliu/models/'
    data_dir = os.path.join(data_dir, opt.data_dir)

    # load generator / projection
    print('Loading network..')
    netC = pg.load_split_classifier(opt.ckc, opt.ckclass, opt.cksplit, evaluate=True, ext=opt.ckcext)
    if opt.informat == 'mat':
        tmpinput = sio.loadmat(opt.input + '.mat',variable_names=['voxels'])
        tmpinput = torch.from_numpy(tmpinput['voxels'])
        #tmpinput = sio.loadmat(opt.input + '.mat',variable_names=['off_volume'])
        #tmpinput = torch.from_numpy(tmpinput['off_volume'])
    else:
        with open(opt.input+'.json','r') as inf: 
            jsonobj = json.load(inf)
            print(jsonobj)
            tmpinput = np.array(jsonobj['data']).reshape(jsonobj['shape'])
            # TODO: for debugging, remove after
            tmpinput = torch.from_numpy(tmpinput)
            sio.savemat(opt.input+'.mat', mdict={'inputs': [], 'voxels': tmpinput.double().unsqueeze_(0).numpy()}) 
            

    inp = torch.Tensor(1,1,64,64,64)
    inpdat = torch.Tensor(1,1,64,64,64)
    criterion = torch.legacy.nn.MSECriterion()
    if opt.gpu > 0:
        netC = netC.cuda()
        inp = inp.cuda()
        inpdat = inpdat.cuda()
        criterion = criterion.cuda()
    netC.evaluate()
    inp = inp.copy_(tmpinput)
    inpfeat = netC.forward(inp)
    inpfeat = inpfeat.clone()

    cache_dir = '/data/jjliu/cache'
    cache_file = os.path.join(cache_dir, opt.data_dir + '_feat.t7')
    cats = os.listdir(data_dir)
    count = 0
    orig_files = []
    for i in range(len(cats)):
        cat_dir = os.path.join(data_dir, cats[i])
        cat_files = os.listdir(cat_dir)
        for j in range(len(cat_files)):
            cat_file = cat_files[j]
            full_cat_file = os.path.join(cat_dir, cat_file)
            #if j < 10:
                #print(full_cat_file)
            orig_files.append(full_cat_file)
            count += 1
    data_feats = torch.Tensor(count, inpfeat.size(1), inpfeat.size(2), inpfeat.size(3), inpfeat.size(4))
    if opt.gpu > 0:
        data_feats = data_feats.cuda()

    if os.path.isfile(cache_file):
        print('loading cached')
        data_feats = torch.load(cache_file)
        if opt.gpu > 0:
            data_feats = data_feats.cuda()
        else:
            data_feats = data_feats.cpu()
    else:
        count = 0
        for i in range(len(orig_files)):
            full_cat_file = orig_files[i]
            print(str(i) + '/' + str(len(cat_files)))
            assert(full_cat_file == orig_files[count])
            if full_cat_file.find('4700.mat') > 0:
                print("FOUND 4700: " + str(count))
            tmpinput2 = sio.loadmat(full_cat_file,variable_names=['off_volume'])
            tmpinput2 = torch.from_numpy(tmpinput2['off_volume'])
            inpdat.zero_()
            inpdat.copy_(tmpinput2)
            featdat = netC.forward(inpdat)
            featdat = featdat.clone()
            data_feats[count].copy_(featdat)
            count += 1
        torch.save(data_feats, cache_file)
    print(count)
    print(data_feats.size())
    assert(count == data_feats.size(0))

    print("Processing dists")
    dists = []
    opt.sample_size = int(opt.sample_size)
    opt.nresults = int(opt.nresults)
    if opt.sample_size == -1:
        sampleindices = range(len(orig_files))
    else:
        sampleindices = random.sample(range(len(orig_files)), int(opt.sample_size))
    
    for a in range(len(sampleindices)):
        i = sampleindices[a]
        dist = criterion.forward(data_feats[i], inpfeat)
        if i == 4048 or i == 4049 or i == 4047: 
            print(i)
            print(orig_files[i])
            print(dist)
        dists.append((i, dist))

    sorted_dists = sorted(dists, key=lambda tup: tup[1])

    print("Saving results..")
    tmplatent = torch.Tensor(1,200,1,1,1)
    for i in range(len(sorted_dists)):
        if i >= opt.nresults:
            break
        index = sorted_dists[i][0]
        orig_file = orig_files[index]
        tmpinput = sio.loadmat(orig_file,variable_names=['off_volume'])
        tmpinput = torch.from_numpy(tmpinput['off_volume'])
        inpdat.copy_(tmpinput)
        inpdat_nd = inpdat.cpu().numpy()
        tmplatent_nd = tmplatent.numpy()
        #sio.savemat(opt.input+'_' + str(i) + '.mat', mdict={'inputs': tmplatent_nd, 'voxels': inpdat_nd}) 
        #torch.save(inpdat, opt.input+'_' + str(i) + '.pic')
        print(os.path.basename(orig_file))
