# Copyright (C) 2018, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import sys
import argparse

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, axis3d, proj3d
import io
from PIL import Image
from sklearn.neighbors import KDTree
import glog as logger
from prepare_data import safe_makedirs
import multiprocessing as multiproc
from functools import partial
import scipy.sparse as sparse

MODLENET40_CLASS_NAMES = [
        'airplane',
        'bathtub',
        'bed',
        'bench',
        'bookshelf',
        'bottle',
        'bowl',
        'car',
        'chair',
        'cone',
        'cup',
        'curtain',
        'desk',
        'door',
        'dresser',
        'flower_pot',
        'glass_box',
        'guitar',
        'keyboard',
        'lamp',
        'laptop',
        'mantel',
        'monitor',
        'night_stand',
        'person',
        'piano',
        'plant',
        'radio',
        'range_hood',
        'sink',
        'sofa',
        'stairs',
        'stool',
        'table',
        'tent',
        'toilet',
        'tv_stand',
        'vase',
        'wardrobe',
        'xbox'
    ]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIS_DIR = os.path.join(BASE_DIR,'visualization')
cmap = plt.get_cmap("Reds")  # "seismic", "gnuplot"

def crop_image(image):
    image_data = np.asarray(image)
    assert(len(image_data.shape)==3)
    image_data_bw = image_data.max(axis=2) if image_data.shape[-1]<=3 else image_data[:,:,3]
    non_empty_columns = np.where(image_data_bw.max(axis=0)>0)[0]
    non_empty_rows = np.where(image_data_bw.max(axis=1)>0)[0]
    cropBox = (min(non_empty_rows), max(non_empty_rows), min(non_empty_columns), max(non_empty_columns))
    image_data_new = image_data[cropBox[0]:cropBox[1]+1, cropBox[2]:cropBox[3]+1 , :]
    return Image.fromarray(image_data_new)

def pts2img(pts, clr, cmap):
    plt.close('all')
    fig = plt.figure()
    fig.set_rasterized(True)
    ax = axes3d.Axes3D(fig)
    pts -= np.mean(pts,axis=0) #demean

    ax.view_init(20,110) # (10,280) for nightstand # ax.view_init(20,110) for general, (20,20) for bookshelf
    ax.set_alpha(255)
    ax.set_aspect('equal')
    min_lim = pts.min()
    max_lim = pts.max()
    ax.set_xlim3d(min_lim,max_lim)
    ax.set_ylim3d(min_lim,max_lim)
    ax.set_zlim3d(min_lim,max_lim)

    if cmap is None and clr is not None:
        assert(np.all(clr.shape==pts.shape))
        ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            c=clr,
            zdir='x',
            s=20,
            edgecolors=(0.5, 0.5, 0.5)  # (0.5,0.5,0.5)
        )
    else:
        if clr is None:
            M = ax.get_proj()
            _,_,clr = proj3d.proj_transform(pts[:,0], pts[:,1], pts[:,2], M)
        clr = (clr-clr.min())/(clr.max()-clr.min()) #normalization
        ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            c=clr,
            zdir='x',
            s=20,
            cmap=cmap,
            edgecolors=(0.5, 0.5, 0.5)  # (0.5,0.5,0.5)
        )

    ax.set_axis_off()
    ax.set_facecolor("white")
    buf = io.BytesIO()
    plt.savefig(
        buf, format='png', transparent=True,
        bbox_inches='tight', pad_inches=0,
        rasterized=True,
        dpi=200
    )
    buf.seek(0)
    im = Image.open(buf)
    im = crop_image(im)
    buf.close()
    return im

def kernel_correlation(pts, G, kernel, sigma):
    npts, ndim = pts.shape
    ret = np.zeros(npts,dtype=np.float32)
    nkps, ndim2 = kernel.shape
    assert(ndim==ndim2)

    # kdt = KDTree(pts, leaf_size=30, metric='euclidean')
    # all_nbs = kdt.query(pts, k=17, return_distance=False)
    assert(isinstance(G, sparse.csr_matrix))

    for i in xrange(npts):
        xi = pts[i,:]
        nbs = G[i].nonzero()[1].tolist() #all_nbs[i,:]
        nnbs = len(nbs)
        ri = 0.0
        for j in xrange(nkps):
            kj = kernel[j,:]
            # rj = 0.0
            for k in xrange(nnbs):
                if nbs[k] == i:
                    logger.warn('this should not happend!')
                    continue
                xk = pts[nbs[k],:]
                vjk = (xk-xi-kj)
                djk = np.dot(vjk,vjk) / sigma
                ri += np.exp(-djk)
        ret[i]=ri/nnbs
    # #normalize
    # ret = (ret-ret.min())/(ret.max()-ret.min())
    return ret

def drawSphere(xCenter, yCenter, zCenter, r):
    #draw sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x=np.cos(u)*np.sin(v)
    y=np.sin(u)*np.sin(v)
    z=np.cos(v)
    # shift and scale sphere
    x = r*x + xCenter
    y = r*y + yCenter
    z = r*z + zCenter
    return (x,y,z)

def kernel2img(kns, min_lim, max_lim, sigma=0.005):
    plt.close('all')
    fig = plt.figure()
    fig.set_rasterized(True)
    ax = axes3d.Axes3D(fig)

    ax.view_init(10, 40)
    ax.set_alpha(255)
    ax.set_aspect('equal')

    for kk in xrange(kns.shape[0]):
        (xs, ys, zs) = drawSphere(kns[kk, 0], kns[kk, 1], kns[kk, 2], r=2*np.sqrt(sigma*0.5)) #use 2*sigma as radius
        ax.plot_surface(xs, ys, zs, color='gray', edgecolors=(0.5, 0.5, 0.5))  # (zs, xs, ys)
        ax.grid(False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.set_xlim3d(min_lim, max_lim)
        ax.set_ylim3d(min_lim, max_lim)
        ax.set_zlim3d(min_lim, max_lim)

    ax.set_facecolor("white")
    buf = io.BytesIO()
    plt.savefig(
        buf, format='png', transparent=True,
        bbox_inches='tight', pad_inches=0,
        rasterized=True,
        dpi=200
    )
    buf.seek(0)
    im = Image.open(buf)
    im = crop_image(im)
    buf.close()
    return im

data_dict = None

def visualize_handcrafted_kernels_core(ith, all_args):
    try:
        global data_dict #large data, shared across multiproc
        handcraft_folder, kernels, args = all_args
        if ith%100==0:
            logger.info('process: %d' % (ith))

        xyz = data_dict['data'][ith]
        lbi = data_dict['label'][ith]
        G = data_dict['graph'][ith]['M']

        obj_folder = os.path.join(handcraft_folder, MODLENET40_CLASS_NAMES[lbi])
        safe_makedirs(obj_folder)

        raw = pts2img(xyz, clr=None, cmap='binary')
        raw.save(os.path.join(obj_folder, '{:04d}_orig.png'.format(ith)))

        for k in xrange(len(kernels)):
            kc_response = kernel_correlation(xyz, G, kernels[k], sigma=args.sigma)
            pts2img(
                xyz,
                kc_response,
                cmap
            ).save(os.path.join(obj_folder, '{:04d}_kc{:02d}.png'.format(ith, k)))
    except KeyboardInterrupt:
        return

def visualize_handcrafted_kernels(args):

    kernels = [
        np.asarray([
            [1,0,0],
            [-1,0,0],
            [2,0,0],
            [-2,0,0]
        ])*0.1,
        np.asarray([
            [0,1,0],
            [0,-1,0],
            [0,2,0],
            [0,-2,0]
        ])*0.1,
        np.asarray([
            [0,0,1],
            [0,0,-1],
            [0,0,2],
            [0,0,-2]
        ])*0.1,
        np.asarray([
            [-1,0,0],
            [0,-1,0],
            [0,0,1],
        ])*0.1,
        np.asarray([
            [0,1,0],
            [0,-1,0],
            [0,0,-1],
        ])*0.1,
        np.asarray([
            [1,0,0],
            [0,1,0],
            [0,-1,0]
        ])*0.2,
        np.asarray([
            [0,1,0],
            [0,-1,0],
            [0,0,-1]
        ])*0.15,
        np.asarray([
            [1,0,0],
            [-1,0,0],
            [0,0,-1],
        ])*0.2,
        np.asarray([
            [1,1,0],
            [1,-1,0],
            [-1,1,0],
            [-1,-1,0]
        ])*0.1,
        np.asarray([
            [0,1,1],
            [0,1,-1],
            [0,-1,1],
            [0,-1,-1]
        ])*0.1
    ]

    #### visualize kernels ####
    min = -0.3
    max = 0.3
    handcraft_folder = os.path.join(VIS_DIR, "hand_crafted_kernels")
    safe_makedirs(handcraft_folder)

    #### visualize kernels ####
    for k in xrange(len(kernels)):
        kni = kernels[k]
        singe_kernel = kernel2img(kni, min, max)
        tname = "hand_kernel_{:02d}.png".format(k)
        singe_kernel.save(os.path.join(handcraft_folder, tname))

    #### visualize kernel responses ####
    pool = multiproc.Pool()
    pool_func = partial(visualize_handcrafted_kernels_core,
                        all_args=(handcraft_folder, kernels, args))
    try:
        pool.map(pool_func, range(args.number))
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
        exit(-1)
    pool.close()

def main(args):
    global data_dict
    data_dict = np.load(args.src).item()
    assert(isinstance(data_dict, dict))
    logger.info('data.shape={}, label.shape={}'.format(data_dict['data'].shape, data_dict['label'].shape))
    args.number = min(args.number, data_dict['data'].shape[0])
    visualize_handcrafted_kernels(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(sys.argv[0])

    parser.add_argument('--src', type=str, default='modelnet/data/modelNet40_test_file_16nn_GM.npy',
                        help="modelnet40 point cloud data file")
    parser.add_argument('--sigma', type=float, default=0.005,
                        help='sigma of kernel')
    parser.add_argument('-n', '--number', type=int, default=100,
                        help='run over the first N objects')

    args = parser.parse_args(sys.argv[1:])
    args.script_folder = os.path.dirname(os.path.abspath(__file__))

    main(args)
