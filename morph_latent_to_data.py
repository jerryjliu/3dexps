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
import numpy as np
import proj_generate as pg


if __name__ == "__main__": 
    # parse args from command line
    parser = OptionParser()
    parser.add_option("--gpu",default=0,help="GPU id, starting from 1. Set it to 0 to run it in CPU mode.")
    parser.add_option('--inplatent',default='morph_in',help='the name of the input file containing the latent vector')
    parser.add_option('--ckp',default='checkpoints_64chair100o_vaen2',help='checkpoint folder of gen model')
    parser.add_option('--ckgen', default='1450',help='checkpoint of the gen model')
    parser.add_option('--ckproj',default='217',help='checkpoint of the projection model')
    parser.add_option('--ckext',default='feat13',help='extension to ckp to specify name of projection folder ( default is none )')
    parser.add_option('--data_dir', default="full_dataset_voxels_64", help="dataset directory of input")
    parser.add_option('--data_cat', default="chair", help="category directory of input")
    parser.add_option('--data_file', help="input data file name (need to specify .mat extension)")
    parser.add_option('--out',default='',help='specify full output file path  (if none put in local output/ folder)')
    parser.add_option('--outformat',default='mat',help='specify format of output (mat, json)')

    
    parser.add_option('--optimize',default=False, help='whether or not to optimize the projection further')
    # only if optimize is true
    parser.add_option('--ckc', default='checkpoints_64class100_5',help='checkpoint folder of classifier feature space')
    parser.add_option('--ckclass', default='200', help='checkpoint of split classifier')
    parser.add_option('--cksplit', default=9, help='split index of classifier')
    parser.add_option('--ckcext', default='net_C', help='extension of the classifier model (typically net_C but can be arbitrary)')
    parser.add_option('--lam', default=0.5, help='[0,1] specifying how much to morph into target file object (1 means completely')
    (opt,args) = parser.parse_args()
    print(opt)
    if opt.gpu > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    data_dir = '/data/jjliu/models/'

    # load generator / projection
    print('Loading network..')
    netG = pg.load_generator(opt.ckp, opt.ckgen, True)
    netP = pg.load_projection(opt.ckp, opt.ckgen, opt.ckproj, True, ext=opt.ckext)

    # load latent input - assume json format
    print('Setting inputs..')
    nz = 200
    with open(os.path.join(data_dir, 'proj_inputs_voxel', opt.inplatent+'.json'),'r') as inf: 
        jsonobj = json.load(inf)
        tmpinput = np.array(jsonobj['data']).reshape(jsonobj['shape'])
        tmpinput = torch.from_numpy(tmpinput)

    # load file object - .mat format
    tmpinput2 = sio.loadmat(os.path.join(data_dir, opt.data_dir, opt.data_cat, opt.data_file),variable_names=['off_volume'])
    tmpinput2 = torch.from_numpy(tmpinput2['off_volume'])
    

    inplatent = torch.Tensor(1,nz,1,1,1)
    inpobj = torch.Tensor(1,1,64,64,64)
    if opt.gpu > 0:
      netG = netG.cuda()
      netP = netP.cuda()
      inplatent = inplatent.cuda()
      inpobj = inpobj.cuda()
    inplatent = inplatent.copy_(tmpinput)
    inpobj = inpobj.copy_(tmpinput2)
    opt.lam = float(opt.lam)

    # propagate input file through P, interpolate between inplatent and objlatent, forward through G
    print('Forward prop')
    objlatent = netP.forward(inpobj)
    print(objlatent)
    inplatent = inplatent.mul((1-opt.lam)).add(objlatent.mul(opt.lam))
    output = netG.forward(inplatent)

    # save result
    print('Saving result')
    print('Output dimensions: ')
    print(output.size())
    fullfname = opt.out
    if opt.gpu > 0:
      inplatent = inplatent.cpu()
      output = output.cpu()
    inplatent_nd = inplatent.numpy()
    output_nd = output.numpy()
    print(output_nd.shape)
    # save input voxels
    sio.savemat(fullfname+'.mat', mdict={'inputs': inplatent_nd, 'voxels': output_nd}) 
    # TODO: SAVE IN READABLE FORAMT
    if opt.outformat == "json": 
        output_nd = output_nd.reshape((64,64,64))
        print(output_nd.shape)
        outdict = {}
        outdict['voxels'] = output_nd.tolist()
        outdict['inputs'] = inplatent_nd.squeeze().tolist()
        with open(fullfname + '.json', 'w') as outf:
            json.dump(outdict, outf)

    print('saving done')
