import torch
import torch.nn as nn
import torch.legacy.nn
import os
from optparse import OptionParser
import scipy.io as sio
from torch.utils.serialization import load_lua
import time
import json
import numpy as np

#checkpoint_gen_path = os.path.join('/data/jjliu/checkpoints/',opt.ckp)
#checkpoint_proj_path = os.path.join('/data/jjliu/checkpoints/',opt.ckp+'_p'+opt.ckgen)
#if opt.ckext != '':
    #checkpoint_proj_path = os.path.join('/data/jjliu/checkpoints/',opt.ckp+'_p'+opt.ckgen+'_'+opt.ckext)


def load_generator(checkpoint_path, epoch, evaluate):
    checkpoint_gen_path = os.path.join('/data/jjliu/checkpoints/',checkpoint_path)
    gen_path = os.path.join(checkpoint_gen_path, 'shapenet101_'+str(epoch)+'_net_G.t7')
    netG = load_lua(gen_path)

    print(netG)
    def zero_conv_bias(m):
        if m.__class__.__name__.find('Convolution') != -1:
            m.bias.zero_()
    netG.apply(zero_conv_bias)
    if evaluate:
        netG.evaluate()
    return netG

def load_projection(checkpoint_path, genEpoch, epoch, evaluate, ext=''):
    checkpoint_proj_path = os.path.join('/data/jjliu/checkpoints/',checkpoint_path+'_p'+str(genEpoch))
    if ext != '':
        checkpoint_proj_path = os.path.join('/data/jjliu/checkpoints/',checkpoint_path+'_p'+str(genEpoch)+'_'+ext)
    proj_path = os.path.join(checkpoint_proj_path, 'shapenet101_'+str(genEpoch)+'_'+str(epoch)+'_net_P.t7')
    netP = load_lua(proj_path)
    print(netP)
    if evaluate:
        netP.evaluate()
    return netP

if __name__ == "__main__": 
    # parse args from command line
    parser = OptionParser()
    parser.add_option("--gpu",default=0,help="GPU id, starting from 1. Set it to 0 to run it in CPU mode.")
    parser.add_option('--input',default='test_chair1',help='the name of the input file')
    parser.add_option('--informat',default='mat',help='format of input object (mat, t7, ndarray)')
    parser.add_option('--ckp',default='checkpoints_64chair_ref',help='checkpoint folder of gen model')
    parser.add_option('--ckgen', default='888',help='checkpoint of the gen model')
    parser.add_option('--ckproj',default='40',help='checkpoint of the projection model')
    parser.add_option('--ckext',default='',help='extension to ckp to specify name of projection folder ( default is none )')
    parser.add_option('--out',default='',help='specify full output file path  (if none put in local output/ folder)')
    parser.add_option('--outformat',default='mat',help='specify format of output (mat, pickle, json)')
    (opt,args) = parser.parse_args()
    print(opt)
    if opt.gpu > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    data_dir = '/data/jjliu/models/proj_inputs_voxel/'

    # load generator / projection
    print('Loading network..')
    netG = load_generator(opt.ckp, opt.ckgen, True)
    netP = load_projection(opt.ckp, opt.ckgen, opt.ckproj, True, ext=opt.ckext)

    # load object input based on input format
    print('Setting inputs..')
    nz = 200
    if opt.informat != 'mat' and opt.informat != 't7' and opt.informat != 'ndarray':
      opt.informat = 'mat'
    if opt.informat == 'mat':
        tmpinput = sio.loadmat(os.path.join(data_dir, opt.input+'.mat'),variable_names=['off_volume'])
        tmpinput = torch.from_numpy(tmpinput['off_volume'])
    elif opt.informat == 't7':
        print('READING T7 file')
        tmpinput = load_lua(os.path.join(data_dir, opt.input+'.t7'))
    elif opt.informat == 'ndarray':
        with open(os.path.join(data_dir, opt.input+'.json'),'r') as inf: 
            jsonobj = json.load(inf)
            tmpinput = np.array(jsonobj['data']).reshape(jsonobj['shape'])
            tmpinput = torch.from_numpy(tmpinput)

    inp = torch.Tensor(1,1,64,64,64)
    if opt.gpu > 0:
      netG = netG.cuda()
      netP = netP.cuda()
      inp = inp.cuda()
    inp = inp.copy_(tmpinput)

    # propagate through P and G and save result
    print('Forward prop')
    latent = netP.forward(inp)
    print(latent)
    output = netG.forward(latent)
    print('Saving result')
    print('Output dimensions: ')
    print(output.size())
    if not os.path.isdir('./output/'):
      os.makedirs('./output/')
    cur_time = int(time.time())
    cur_times = '' + str(cur_time)
    fname = 'proj_' + cur_times + '_' + opt.ckgen + '_' + opt.ckproj
    if opt.out is None or opt.out == '':
      fullfname = os.path.join('./output', fname)
    else:
      fullfname = opt.out
    if opt.gpu > 0:
      latent = latent.cpu()
      output = output.cpu()
    latent_nd = latent.numpy()
    output_nd = output.numpy()
    print(output_nd.shape)
    # save input voxels
    inp_nd = inp.cpu().numpy()
    sio.savemat(os.path.join(data_dir, opt.input+'.mat'), mdict={'inputs': latent_nd, 'voxels': inp_nd}) 
    sio.savemat(fullfname+'.mat', mdict={'inputs': latent_nd, 'voxels': output_nd}) 
    # TODO: SAVE IN READABLE FORAMT
    if opt.outformat == "pickle":
        torch.save(output, fullfname + '.pic')
        #torch.save(fullfname + '.pic', output)
    elif opt.outformat == "json": 
        output_nd = output_nd.reshape((64,64,64))
        print(output_nd.shape)
        with open(fullfname + '.json', 'w') as outf:
            json.dump(output_nd.tolist(), outf)

    print('saving done')
