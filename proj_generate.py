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

def load_split_classifier(checkpoint_path, classEpoch, splitIndex, evaluate=True, ext='net_C'):
    checkpoint_proj_path = os.path.join('/data/jjliu/checkpoints/',checkpoint_path)
    class_path = os.path.join(checkpoint_proj_path, 'shapenet101_'+str(classEpoch)+'_' + ext + '_split'+str(splitIndex)+'.t7')
    print(class_path)
    netC = load_lua(class_path)
    print(netC)
    if evaluate:
        netC.evaluate()
    return netC

def project_input(netP, netG, inp):
    latent = netP.forward(inp)
    output = netG.forward(latent)
    return output, latent


def optimize_latent(netG, netC, criterion, startLatent, inputObj, steps=200, nz=200):
    optimStateC = {}
    optimStateC['learningRate'] = 0.1
    latent = startLatent.view(-1)
    print(startLatent.size())
    print('optimizing latent..')

    prevErr = None
    errL = None

    def fLx(latent):
        latent = latent.view(1,nz,1,1,1)
        out = netG.forward(latent)
        outfeat = netC.forward(out)
        outfeat = outfeat.clone()
        reffeat = netC.forward(inputObj)
        errL = criterion.forward(outfeat, reffeat)
        print(errL)
        df_dc = criterion.backward(outfeat, reffeat)
        df_do = netC.updateGradInput(out, df_dc)
        df_dl = netG.updateGradInput(latent, df_do)
        return errL, df_dl

    for i in range(steps):
        print('Optimizing latent step: ' + str(i))
        _, errL = torch.legacy.optim.sgd(fLx, latent, optimStateC)
        if prevErr is not None and errL is not None and prevErr < errL and i > 60:
            break
        if i % 4 == 0:
            prevErr = errL

    return latent.view(1,nz,1,1,1)


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

    
    parser.add_option('--optimize',default=False, help='whether or not to optimize the projection further')
    # only if optimize is true
    parser.add_option('--ckc', default='checkpoints_64class100_5',help='checkpoint folder of classifier feature space')
    parser.add_option('--ckclass', default='200', help='checkpoint of split classifier')
    parser.add_option('--cksplit', default=9, help='split index of classifier')
    parser.add_option('--ckcext', default='net_C', help='extension of the classifier model (typically net_C but can be arbitrary)')
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
    # if optimize is true, then optimize further
    if opt.optimize:
        netG.training()
        netC = load_split_classifier(opt.ckc, opt.ckclass, opt.cksplit, evaluate=False, ext=opt.ckcext)
        mseCriterion = torch.legacy.nn.MSECriterion()
        if opt.gpu > 0:
            netC = netC.cuda()
            mseCriterion = mseCriterion.cuda()
        latent = optimize_latent(netG, netC, mseCriterion, latent, inp)
        netG.evaluate()

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
    sio.savemat(fullfname+'.mat', mdict={'inputs': latent_nd, 'voxels': output_nd}) 
    # TODO: SAVE IN READABLE FORAMT
    if opt.outformat == "pickle":
        torch.save(output, fullfname + '.pic')
        #torch.save(fullfname + '.pic', output)
    elif opt.outformat == "json": 
        sio.savemat(os.path.join(data_dir, opt.input+'.mat'), mdict={'inputs': latent_nd, 'voxels': inp_nd}) 
        output_nd = output_nd.reshape((64,64,64))
        print(output_nd.shape)
        outdict = {}
        outdict['voxels'] = output_nd.tolist()
        outdict['inputs'] = latent_nd.squeeze().tolist()
        with open(fullfname + '.json', 'w') as outf:
            json.dump(outdict, outf)

    print('saving done')
