import torch
import torch.nn as nn
import torch.legacy.nn
import os
from optparse import OptionParser
import scipy.io as sio
from torch.utils.serialization import load_lua
import time

parser = OptionParser()
parser.add_option("--gpu",default=0,help="GPU id, starting from 1. Set it to 0 to run it in CPU mode.")
parser.add_option('--input',default='test_chair1',help='the name of the input file')
parser.add_option('--informat',default='mat',help='format of input object (mat, t7)')
parser.add_option('--ckp',default='checkpoints_64chair_ref',help='checkpoint folder of gen model')
parser.add_option('--ckgen', default='888',help='checkpoint of the gen model')
parser.add_option('--ckproj',default='40',help='checkpoint of the projection model')
parser.add_option('--ckext',default='',help='extension to ckp to specify name of projection folder ( default is none )')
parser.add_option('--out',default='',help='specify full output file path  (if none put in local output/ folder)')
parser.add_option('--outformat',default='mat',help='specify format of output (mat, pickle)')

(opt,args) = parser.parse_args()
print(opt)

if opt.gpu > 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

data_dir = '/data/jjliu/models/proj_inputs_voxel/'
checkpoint_gen_path = os.path.join('/data/jjliu/checkpoints/',opt.ckp)
checkpoint_proj_path = os.path.join('/data/jjliu/checkpoints/',opt.ckp+'_p'+opt.ckgen)
if opt.ckext != '':
    checkpoint_proj_path = os.path.join('/data/jjliu/checkpoints/',opt.ckp+'_p'+opt.ckgen+'_'+opt.ckext)

print('Loading network..')
gen_path = os.path.join(checkpoint_gen_path, 'shapenet101_'+opt.ckgen+'_net_G.t7')
proj_path = os.path.join(checkpoint_proj_path, 'shapenet101_'+opt.ckgen+'_'+opt.ckproj+'_net_P.t7')
print(proj_path)
netG = load_lua(gen_path)
netP = load_lua(proj_path)
print(netG)
print(netP)

if opt.gpu == 0:
  netG = netG.double()
  netP = netP.double()

nz = 200
# TODO: remove convolution bias for netG
print(netG.__class__)
def zero_conv_bias(m):
    if m.__class__.__name__.find('Convolution') != -1:
        m.bias.zero_()
netG.apply(zero_conv_bias)
#netG.train(False)
#netP.train(False)
netG.evaluate()
netP.evaluate()

print('Setting inputs..')
if opt.informat != 'mat' and opt.informat != 't7':
  opt.informat = 'mat'
if opt.informat == 'mat':
    tmpinput = sio.loadmat(os.path.join(data_dir, opt.input+'.mat'),variable_names=['off_volume'])
else:
    tmpinput2 = load_lua(os.path.join(data_dir, opt.input+'.t7'))
inp = torch.Tensor(1,1,64,64,64)
if opt.gpu > 0:
  netG = netG.cuda()
  netP = netP.cuda()
  inp = inp.cuda()

tmpinput = torch.from_numpy(tmpinput['off_volume'])

inp = inp.copy_(tmpinput)
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

if opt.outformat != 'mat' and opt.outformat != 't7':
  opt.outformat = 'mat'

latent = latent.numpy()
output = output.numpy()
sio.savemat(fullfname+'.mat', mdict={'inputs': latent, 'voxels': output}) 
# TODO: SAVE IN READABLE FORAMT
if opt.outformat == "pickle":
  torch.save(fullfname + '.pic', output)

print('saving done')
