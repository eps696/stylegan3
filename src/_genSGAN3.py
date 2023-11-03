import os
import os.path as osp
import argparse
import random
import numpy as np
from imageio import imsave

import torch

import dnnlib
import legacy
from torch_utils import misc

from util.utilgan import latent_anima, basename, img_read
from util.progress_bar import progbar

desc = "Customized StyleGAN3 on PyTorch"
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('-o', '--out_dir', default='_out', help='output directory')
parser.add_argument('-m', '--model', default='models/ffhq-1024.pkl', help='path to pkl checkpoint file')
parser.add_argument('-l', '--labels', type=int, default=None, help='labels/categories for conditioning')
# custom
parser.add_argument('-s', '--size', default=None, help='Output resolution')
parser.add_argument('-sc', '--scale_type', default='pad', help="may include pad, side, symm (also centr, fit)")
parser.add_argument('-lm', '--latmask', default=None, help='external mask file (or directory) for multi latent blending')
parser.add_argument('-n', '--nXY', default='1-1', help='multi latent frame split count by X (width) and Y (height)')
parser.add_argument('--splitfine', type=float, default=0, help='multi latent frame split edge sharpness (0 = smooth, higher => finer)')
parser.add_argument('--splitmax', type=int, default=None, help='max count of latents for frame splits (to avoid OOM)')
parser.add_argument('--trunc', type=float, default=0.8, help='truncation psi 0..1 (lower = stable, higher = various)')
parser.add_argument('--save_lat', action='store_true', help='save latent vectors to file')
parser.add_argument('-v', '--verbose', action='store_true')
# animation
parser.add_argument('-f', '--frames', default='200-25', help='total frames to generate, length of interpolation step')
parser.add_argument("--cubic", action='store_true', help="use cubic splines for smoothing")
parser.add_argument("--gauss", action='store_true', help="use Gaussian smoothing")
# transform SG3
parser.add_argument('-at', "--anim_trans", action='store_true', help="add translation animation")
parser.add_argument('-ar', "--anim_rot", action='store_true', help="add rotation animation")
parser.add_argument('-sb', '--shiftbase', type=float, default=0., help='Shift to the tile center?')
parser.add_argument('-sm', '--shiftmax',  type=float, default=0., help='Random walk around tile center')
a = parser.parse_args()

if a.size is not None: 
    a.size = [int(s) for s in a.size.split('-')][::-1]
    if len(a.size) == 1: a.size = a.size * 2
[a.frames, a.fstep] = [int(s) for s in a.frames.split('-')]

def checkout(output, i):
    ext = 'png' if output.shape[3]==4 else 'jpg'
    filename = osp.join(a.out_dir, "%06d.%s" % (i,ext))
    imsave(filename, output[0], quality=95)
    
def generate():
    os.makedirs(a.out_dir, exist_ok=True)
    np.random.seed(seed=696)
    device = torch.device('cuda')

    # setup generator
    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.verbose = a.verbose
    Gs_kwargs.size = a.size
    Gs_kwargs.scale_type = a.scale_type

    # mask/blend latents with external latmask or by splitting the frame
    if a.latmask is None:
        nHW = [int(s) for s in a.nXY.split('-')][::-1]
        assert len(nHW)==2, ' Wrong count nXY: %d (must be 2)' % len(nHW)
        n_mult = nHW[0] * nHW[1]
        if a.splitmax is not None: n_mult = min(n_mult, a.splitmax)
        Gs_kwargs.countHW = nHW
        Gs_kwargs.splitfine = a.splitfine
        if a.splitmax is not None: Gs_kwargs.splitmax = a.splitmax
        if a.verbose is True and n_mult > 1: print(' Latent blending w/split frame %d x %d' % (nHW[1], nHW[0]))
        lmask = [None]
    
    else:
        n_mult = 2
        nHW = [1,1]
        if osp.isfile(a.latmask): # single file
            lmask = np.asarray([[img_read(a.latmask)[:,:,0] / 255.]]) # [1,1,h,w]
        elif osp.isdir(a.latmask): # directory with frame sequence
            lmask = np.expand_dims(np.asarray([img_read(f)[:,:,0] / 255. for f in img_list(a.latmask)]), 1) # [n,1,h,w]
        else:
            print(' !! Blending mask not found:', a.latmask); exit(1)
        if a.verbose is True: print(' Latent blending with mask', a.latmask, lmask.shape)
        lmask = np.concatenate((lmask, 1 - lmask), 1) # [n,2,h,w]
        lmask = torch.from_numpy(lmask).to(device)
    
    # load base or custom network
    pkl_name = osp.splitext(a.model)[0]
    if '.pkl' in a.model.lower():
        custom = False
        print(' .. Gs from pkl ..', basename(a.model))
    else:
        custom = True
        print(' .. Gs custom ..', basename(a.model))
    rot = True if ('-r-' in a.model.lower() or 'sg3r-' in a.model.lower()) else False
    with dnnlib.util.open_url(pkl_name + '.pkl') as f:
        Gs = legacy.load_network_pkl(f, custom=custom, rot=rot, **Gs_kwargs)['G_ema'].to(device) # type: ignore

    if a.size is None: a.size = [Gs.img_resolution] * 2

    if a.verbose is True: print(' making timeline..')
    latents = latent_anima((n_mult, Gs.z_dim), a.frames, a.fstep, cubic=a.cubic, gauss=a.gauss, verbose=False) # [frm,X,512]
    print(' latents', latents.shape)
    latents = torch.from_numpy(latents).to(device)
    frame_count = latents.shape[0]
    
    # labels / conditions
    label_size = Gs.c_dim
    if label_size > 0:
        labels = torch.zeros((frame_count, n_mult, label_size), device=device) # [frm,X,lbl]
        if a.labels is None:
            label_ids = []
            for i in range(n_mult):
                label_ids.append(random.randint(0, label_size-1))
        else:
            label_ids = [int(x) for x in a.labels.split('-')]
            label_ids = label_ids[:n_mult] # ensure we have enough labels
        for i, l in enumerate(label_ids):
            labels[:,i,l] = 1
    else:
        labels = [None]

    # NEW SG3
    if hasattr(Gs.synthesis, 'input'): # SG3
        if a.anim_trans is True:
            hw_centers = [np.linspace(-1+1/n, 1-1/n, n) for n in nHW]
            yy,xx = np.meshgrid(*hw_centers)
            xscale = [s / Gs.img_resolution for s in a.size]
            hw_centers = np.dstack((yy.flatten()[:n_mult], xx.flatten()[:n_mult])) * xscale * 0.5 * a.shiftbase
            hw_scales = np.array([2. / n for n in nHW]) * a.shiftmax
            shifts = latent_anima((n_mult, 2), a.frames, a.fstep, uniform=True, cubic=a.cubic, gauss=a.gauss, verbose=False) # [frm,X,2]
            shifts = hw_centers + (shifts - 0.5) * hw_scales
        else:
            shifts = np.zeros((1, n_mult, 2))
        if a.anim_rot is True:
            angles = latent_anima((n_mult, 1), a.frames, a.frames//4, uniform=True, cubic=a.cubic, gauss=a.gauss, verbose=False) # [frm,X,1]
            angles = (angles - 0.5) * 180.
        else:
            angles = np.zeros((1, n_mult, 1))
        shifts = torch.from_numpy(shifts).to(device)
        angles = torch.from_numpy(angles).to(device)
        trans_params = list(zip(shifts, angles))
        
    # warm up
    if custom:
        if hasattr(Gs.synthesis, 'input'): # SG3
            _ = Gs(latents[0], labels[0], lmask[0], trans_params[0], noise_mode='const')
        else: # SG2
            _ = Gs(latents[0], labels[0], lmask[0], noise_mode='const')
    else:
        _ = Gs(latents[0], labels[0], noise_mode='const')
    
    # generate images from latent timeline
    pbar = progbar(frame_count)
    for i in range(frame_count):
    
        latent  = latents[i] # [X,512]
        label   = labels[i % len(labels)]
        latmask = lmask[i % len(lmask)] # [X,h,w] or None
        if hasattr(Gs.synthesis, 'input'): # SG3
            trans_param = trans_params[i % len(trans_params)]

        # generate multi-latent result
        if custom:
            if hasattr(Gs.synthesis, 'input'): # SG3
                output = Gs(latent, label, latmask, trans_param, truncation_psi=a.trunc, noise_mode='const')
            else: # SG2
                output = Gs(latent, label, latmask, truncation_psi=a.trunc, noise_mode='const')
        else:
            output = Gs(latent, label, truncation_psi=a.trunc, noise_mode='const')
        output = (output.permute(0,2,3,1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()

        # save image
        checkout(output, i)
        pbar.upd()


    if a.save_lat is True:
        latents = latents.squeeze(1) # [frm,512]
        if a.size is None: a.size = ['']*2
        filename = '{}-{}-{}.npy'.format(basename(a.model), a.size[1], a.size[0])
        filename = osp.join(osp.dirname(a.out_dir), filename)
        latents = latents.cpu().numpy()
        np.save(filename, latents)
        print('saved latents', latents.shape, 'to', filename)


if __name__ == '__main__':
    generate()
