# Copyright (C) 2018, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import sys
import numpy as np
import h5py
import argparse
import scipy.sparse
from sklearn.neighbors import KDTree
import multiprocessing as multiproc
from functools import partial
import glog as logger
from copy import deepcopy
import errno
import gdown #https://github.com/wkentaro/gdown

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def safe_makedirs(d):
    if not os.path.exists(d):
        try:
            os.makedirs(d)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            pass

def get_modelnet40_data_npy_name():
    folder = 'modelnet'
    DATA_DIR = os.path.join(BASE_DIR, folder, 'data')
    safe_makedirs(DATA_DIR)
    train_npy = os.path.join(DATA_DIR, 'modelNet40_train_file.npy')
    test_npy = os.path.join(DATA_DIR, 'modelNet40_test_file.npy')
    return train_npy, test_npy, DATA_DIR

def download_modelnet40_data(num_points):
    assert(0<=num_points<=2048)
    def h5_to_npy(h5files, npy_fname, num_points):
        all_data = []
        all_label = []
        for h5 in h5files:
            f = h5py.File(h5)
            data = f['data'][:]
            data = data[:,:num_points,:]
            label = f['label'][:]
            all_data.append(data)
            all_label.append(label)
        all_data = np.concatenate(all_data, axis=0) #<BxNx3>
        all_label = np.concatenate(all_label, axis=0)[:,0] #<B,>
        np.save(npy_fname, {'data':all_data, 'label':all_label})
        logger.info('saved: '+npy_fname)

    train_npy, test_npy, DATA_DIR = get_modelnet40_data_npy_name()
    remote_name = 'modelnet40_ply_hdf5_2048'

    if not os.path.exists(train_npy):
        www = 'https://shapenet.cs.stanford.edu/media/{}.zip'.format(remote_name)
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))

        h5folder = os.path.join(DATA_DIR, remote_name)
        h5files = [f for f in os.listdir(h5folder) if f.endswith('.h5')]
        test_files = sorted([os.path.join(h5folder, f) for f in h5files if f.startswith('ply_data_test')])
        train_files = sorted([os.path.join(h5folder, f) for f in h5files if f.startswith('ply_data_train')])
        h5_to_npy(h5files=test_files, npy_fname=test_npy, num_points=num_points)
        h5_to_npy(h5files=train_files, npy_fname=train_npy, num_points=num_points)

        os.system('rm -r %s' % h5folder)

    td = np.load(train_npy).item()
    assert(isinstance(td, dict))
    assert(td.has_key('data'))
    assert(td.has_key('label'))
    assert(td['data'].shape==(9840,num_points,3))
    assert(td['label'].shape==(9840,))

    td = np.load(test_npy).item()
    assert(isinstance(td, dict))
    assert(td.has_key('data'))
    assert(td.has_key('label'))
    assert(td['data'].shape==(2468,num_points,3))
    assert(td['label'].shape==(2468,))

def get_shapenet_data_npy_name():
    folder = 'shapenet'
    DATA_DIR = os.path.join(BASE_DIR, folder, 'data')
    safe_makedirs(DATA_DIR)
    train_npy = os.path.join(DATA_DIR, 'shapenet_train_file.npy')
    test_npy = os.path.join(DATA_DIR, 'shapenet_test_file.npy')
    return train_npy, test_npy, DATA_DIR

def download_shapenet_data(num_points):
    assert(0<=num_points<=2048)
    def h5_to_npy(h5files, npy_fname, num_points):
        all_data = []
        all_label = []
        all_seglabel = []
        for h5 in h5files:
            f = h5py.File(h5)
            data = f['data'][:]
            data = data[:,:num_points,:]
            label = f['label'][:]
            seg = f['pid'][:]
            seg = seg[:,:num_points]
            all_data.append(data)
            all_label.append(label)
            all_seglabel.append(seg)
        all_data = np.concatenate(all_data, axis=0) #<BxNx3>
        all_label = np.concatenate(all_label, axis=0)[:,0] #<B,>
        all_seglabel = np.concatenate(all_seglabel, axis=0) #<BxN>
        np.save(npy_fname, {'data':all_data, 'label':all_label, 'seg_label': all_seglabel})
        logger.info('saved: '+npy_fname)

    train_npy, test_npy, DATA_DIR = get_shapenet_data_npy_name()

    if not os.path.exists(train_npy):
        www = 'https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www, zipfile))
        os.system('mv hdf5_data %s' % (DATA_DIR))
        os.system('rm %s' % (zipfile))

        h5folder = os.path.join(DATA_DIR, 'hdf5_data')
        h5files = [f for f in os.listdir(h5folder) if f.endswith('.h5')]
        test_files = sorted([os.path.join(h5folder, f) for f in h5files if f.startswith('ply_data_test')])
        val_files = sorted([os.path.join(h5folder, f) for f in h5files if f.startswith('ply_data_val')])
        train_files = sorted([os.path.join(h5folder, f) for f in h5files if f.startswith('ply_data_train')])
        train_files += val_files #note: following pointnet++'s protocal
        h5_to_npy(h5files=test_files, npy_fname=test_npy, num_points=num_points)
        h5_to_npy(h5files=train_files, npy_fname=train_npy, num_points=num_points)

        os.system('rm -r %s' % h5folder)

    td = np.load(train_npy).item()
    assert(isinstance(td, dict))
    assert(td.has_key('data'))
    assert(td.has_key('label'))
    assert(td.has_key('seg_label'))
    assert(td['data'].shape==(14007,num_points,3))
    assert(td['label'].shape==(14007,))
    assert(td['seg_label'].shape==(14007,num_points))

    td = np.load(test_npy).item()
    assert(isinstance(td, dict))
    assert(td.has_key('data'))
    assert(td.has_key('label'))
    assert(td.has_key('seg_label'))
    assert(td['data'].shape==(2874,num_points,3))
    assert(td['label'].shape==(2874,))
    assert(td['seg_label'].shape==(2874,num_points))

def get_modelnet10_data_npy_name():
    folder = 'modelnet'
    DATA_DIR = os.path.join(BASE_DIR, folder, 'data')
    safe_makedirs(DATA_DIR)
    train_npy = os.path.join(DATA_DIR, 'modelNet10_train.npy')
    test_npy = os.path.join(DATA_DIR, 'modelNet10_test.npy')
    return train_npy, test_npy, DATA_DIR

def download_modelnet10_data():
    num_points = 1024
    train_npy, test_npy, DATA_DIR = get_modelnet10_data_npy_name()

    if not os.path.exists(train_npy):
        www = '"https://drive.google.com/uc?export=download&id=19ktNMgr51vX30vBjoUjxYi4EJoWOfr4m"'
        zipfile = 'modelNet10.zip'
        os.system('wget %s -O %s; unzip %s -d %s' % (www, zipfile, zipfile, DATA_DIR))
        os.system('rm %s' % (zipfile))

    td = np.load(train_npy).item()
    assert(isinstance(td, dict))
    assert(td.has_key('data'))
    assert(td.has_key('label'))
    assert(td['data'].shape==(3991,num_points,3))
    assert(td['label'].shape==(3991,))

    td = np.load(test_npy).item()
    assert(isinstance(td, dict))
    assert(td.has_key('data'))
    assert(td.has_key('label'))
    assert(td['data'].shape==(908,num_points,3))
    assert(td['label'].shape==(908,))

##########################################################################################

def edges2A(edges, n_nodes, mode='P', sparse_mat_type=scipy.sparse.csr_matrix):
    '''
    note: assume no (i,i)-like edge
    edges: <2xE>
    '''
    edges = np.array(edges).astype(int)

    data_D = np.zeros(n_nodes, dtype=np.float32)
    for d in xrange(n_nodes):
        data_D[ d ] = len(np.where(edges[0] == d)[0])

    if mode.upper() == 'M':  # 'M' means max pooling, which use the same graph matrix as the adjacency matrix
        data = np.ones(edges[0].shape[0], dtype=np.int32)
    elif mode.upper() == 'P':
        data = 1. / data_D[ edges[0] ]
    else:
        raise NotImplementedError("edges2A with unknown mode=" + mode)

    return sparse_mat_type((data, edges), shape=(n_nodes, n_nodes))

def knn_search(data, knn, metric="euclidean", return_distance=False, symmetric=True):
    """
    Args:
      data: Nx3
      knn: default=4
      metric:
      return_distance: bool

    Returns: edges and weighted+normalized adj matrix if return_distance is True, otherwise return edges
    """
    assert(knn>0)
    n_data_i = data.shape[0]
    kdt = KDTree(data, leaf_size=30, metric=metric)

    if return_distance:
        nbs = kdt.query(data, k=knn+1, return_distance=True)
        adjdict = dict()
        # wadj = np.zeros((n_data_i, n_data_i), dtype=np.float32)
        for i in xrange(n_data_i):
            # nbsd = nbs[0][i]
            nbsi = nbs[1][i]
            for j in xrange(knn):
                if symmetric:
                    adjdict[(i, nbsi[j+1])] = 1
                    adjdict[(nbsi[j+1], i)] = 1
                    # wadj[i, nbsi[j + 1]] = 1.0 / nbsd[j + 1]
                    # wadj[nbsi[j + 1], i] = 1.0 / nbsd[j + 1]
                else:
                    adjdict[(i, nbsi[j+1])] = 1
                    # wadj[i, nbsi[j + 1]] = 1.0 / nbsd[j + 1]
        edges = np.array(adjdict.keys(), dtype=int).T
        return edges, nbs[0] #, wadj
    else:
        nbs = kdt.query(data, k=knn+1, return_distance=False)  # sort_results=True (default)
        adjdict = dict()
        for i in xrange(n_data_i):
            nbsi = nbs[i]
            for j in xrange(knn):
                if symmetric:
                    adjdict[(i, nbsi[j+1])] = 1  # exclude the node itself as its neighbors
                    adjdict[(nbsi[j+1], i)] = 1  # make symmetric
                else:
                    adjdict[(i, nbsi[j+1])] = 1
        edges = np.array(adjdict.keys(), dtype=int).T
        return edges

def build_graph_core(ith, args, data):
    xyi = data[ith, ...] # 1024x3
    n_data_i = xyi.shape[0]
    edges, nbsd = knn_search(xyi, knn=args.knn, metric=args.metric, return_distance=True)
    ith_graph = edges2A(edges, n_data_i, args.mode, sparse_mat_type=scipy.sparse.csr_matrix)
    nbsd=np.asarray(nbsd)[:,1:]
    nbsd=np.reshape(nbsd,-1)

    if ith % 500 == 0:
        logger.info('{} processed: {}'.format(args.flag, ith))

    return ith, ith_graph, nbsd

def build_graphs(data_dict, args):
    '''
    Build graphs based on mode of all data.
    Data and graphs are saved in args.src (path).
    '''
    total_num_data = data_dict['label'].shape[0]

    if args.shuffle:
        idx = np.arange(total_num_data)
        np.random.shuffle(idx)
        data_dict['data'] = data_dict['data'][idx]
        data_dict['label'] = data_dict['label'][idx]
        if data_dict.has_key('seg_label'):
            data_dict['seg_label'] = data_dict['seg_label'][idx]

    graphs = [{} for i in range(total_num_data)]
    all_nbs_dist = []

    #parallel version
    pool = multiproc.Pool(multiproc.cpu_count())
    pool_func = partial(build_graph_core, args=args, data=data_dict['data'])
    rets = pool.map(pool_func, range(total_num_data))
    pool.close()
    for ret in rets:
        ith, ith_graph, nbsd = ret
        graphs[ith][args.mode] = ith_graph
        all_nbs_dist.append(nbsd)

    # #sequential version
    # for ith in xrange(total_num_data):
    #     if ith % 100 == 0:
    #         logger.info('save: {}'.format(ith))
    #
    #     xyi = data_dict['data'][ith, ...] # 1024x3
    #     n_data_i = xyi.shape[0]
    #     edges, nbsd = knn_search(xyi, knn=args.knn, metric=args.metric, return_distance=True)
    #     graphs[ith][args.mode] = edges2A(edges, n_data_i, args.mode, sparse_mat_type=scipy.sparse.csr_matrix)
    #     nbsd=np.asarray(nbsd)[:,1:]
    #     nbsd=np.reshape(nbsd,-1)
    #     all_nbs_dist.append(nbsd)

    all_nbs_dist = np.stack(all_nbs_dist)
    mean_nbs_dist = all_nbs_dist.mean()
    std_nbs_dist = all_nbs_dist.std()

    logger.info('{}: neighbor distance: mean={:f}, std={:f}'.format(args.flag, mean_nbs_dist, std_nbs_dist))
    data_dict.update({'graph': graphs, 'mean_nbs_dist':mean_nbs_dist, 'std_nbs_dist':std_nbs_dist})
    np.save(args.dst, data_dict)
    logger.info('saved: '+args.dst)

##########################################################################################

def load_data(data_file):
    data_dict = np.load(data_file).item()
    assert(isinstance(data_dict, dict))
    total_data = data_dict['data']
    total_labels = data_dict['label']
    logger.info('all data size:')
    logger.info('data: '+str(total_data.shape))
    logger.info('label: '+str(total_labels.shape))
    if data_dict.has_key('seg_label'):
        logger.info('seg_label: '+str(data_dict['seg_label'].shape))

    concat = np.concatenate(total_data, axis=0)
    flag = os.path.basename(data_file)
    logger.info('{}: all data points locates within:'.format(flag))
    logger.info('{}: min: {}'.format(flag, tuple(concat.min(axis=0))))
    logger.info('{}: max: {}'.format(flag, tuple(concat.max(axis=0))))

    return data_dict

def raw_npy_to_graph_npy(raw_npy, args):
    fname_prefix = os.path.splitext(raw_npy)[0]
    return '{}_{}nn_G{}.npy'.format(fname_prefix, args.knn, args.mode)

def process_one_npy(args):
    args.flag = os.path.basename(args.src)
    args.dst = raw_npy_to_graph_npy(args.src, args)
    if os.path.exists(args.dst):
        logger.info('{} existed already.'.format(args.dst))
        return
    logger.info('{} ==(build_graphs)==> {}'.format(args.src, args.dst))
    data_dict = load_data(args.src)
    build_graphs(data_dict, args)

def run_all_processes(all_p):
    try:
        for p in all_p:
            p.start()
        for p in all_p:
            p.join()
    except KeyboardInterrupt:
        for p in all_p:
            if p.is_alive():
                p.terminate()
            p.join()
        exit(-1)

def process_all_npy(all_npy, args):
    all_p = []
    for the_npy in all_npy:
        the_args = deepcopy(args)
        the_args.src = the_npy
        p = multiproc.Process(target=process_one_npy, args=(the_args,))
        all_p.append(p)
    run_all_processes(all_p)

##########################################################################################

def download_modelnet_experiments_from_google_drive_and_unzip():
    www = 'https://drive.google.com/uc?id=12u-eH3Ag_K3eydUIbaYzreQikCNgJtGl'
    zipfile = 'modelnet_experiments.tar.gz'
    dst_folder = os.path.join(BASE_DIR, 'modelnet')
    gdown.download(url=www,output=zipfile,quiet=False)
    os.system('tar xvzf %s -C %s' % (zipfile, dst_folder))
    os.system('rm %s' % (zipfile))

def download_shapenet_experiments_from_google_drive_and_unzip():
    www = 'https://drive.google.com/uc?id=11bZYlTOpwsdXqQzD1NuXiRpTD6x1Ewup'
    zipfile = 'shapenet_experiments.tar.gz'
    dst_folder = os.path.join(BASE_DIR, 'shapenet')
    gdown.download(url=www,output=zipfile,quiet=False)
    os.system('tar xvzf %s -C %s' % (zipfile, dst_folder))
    os.system('rm %s' % (zipfile))

def download_tar_from_google_drive_and_unzip(www, zipfile, train_npy, test_npy, DATA_DIR, args, flag):
    train_npy = raw_npy_to_graph_npy(train_npy, args)
    test_npy = raw_npy_to_graph_npy(test_npy, args)
    if not os.path.exists(train_npy):
        gdown.download(url=www,output=zipfile,quiet=False)
        os.system('tar xvzf %s -C %s' % (zipfile, DATA_DIR))
        os.system('rm %s' % (zipfile))
        assert(os.path.exists(train_npy))
        assert(os.path.exists(test_npy))
    else:
        logger.info('{} data files exited!'.format(flag))

def prepare_modelnet40(args):
    logger.info('preparing modelnet40')
    train_npy, test_npy, DATA_DIR = get_modelnet40_data_npy_name()
    args.knn=16

    if args.pts_mn40==1024 and args.mode=='M' and args.knn==16 and not args.regenerate:
        download_tar_from_google_drive_and_unzip(
            www='https://drive.google.com/uc?id=1P5N84BSNbMYejqQv8eu31BAP0Dl5DDE7',
            zipfile='modelNet40.tar.gz',
            flag='modelnet40',
            train_npy=train_npy, test_npy=test_npy,DATA_DIR=DATA_DIR,args=args
        )
    else:
        download_modelnet40_data(num_points=args.pts_mn40)
        process_all_npy(
            [train_npy, test_npy],
            args=args
        )
    logger.info('modelnet40 done!')

def prepare_modelnet10(args):
    logger.info('preparing modelnet10')
    train_npy, test_npy, DATA_DIR = get_modelnet10_data_npy_name()
    args.knn=16

    if args.pts_mn10==1024 and args.mode=='M' and args.knn==16 and not args.regenerate:
        download_tar_from_google_drive_and_unzip(
            www='https://drive.google.com/uc?id=1fcEyTMtZTBavXTG3-by3aqQNZQnxNOBO',
            zipfile='modelNet10.tar.gz',
            flag='modelnet10',
            train_npy=train_npy, test_npy=test_npy,DATA_DIR=DATA_DIR,args=args
        )
    else:
        download_modelnet10_data()
        process_all_npy(
            [train_npy, test_npy],
            args=args
        )
    logger.info('modelnet10 done!')

def prepare_shapenet(args):
    logger.info('preparing shapenet')
    train_npy, test_npy, DATA_DIR = get_shapenet_data_npy_name()
    args.knn=18

    if args.pts_shapenet==2048 and args.mode=='M' and args.knn==18 and not args.regenerate:
        download_tar_from_google_drive_and_unzip(
            www='https://drive.google.com/uc?id=1zRZsOzxGPSskY6AsaumNPJVg9eDnf9kN',
            zipfile='shapenet_2048_18nn_GM.tar.gz',
            flag='shapenet',
            train_npy=train_npy, test_npy=test_npy,DATA_DIR=DATA_DIR,args=args
        )
    else:
        download_shapenet_data(num_points=2048)
        process_all_npy(
            [train_npy, test_npy],
            args=args
        )
    logger.info('shapenet done!')

def main(args):
    assert(len(args.mode)==1)
    all_p = []
    all_p.append(
        multiproc.Process(target=prepare_modelnet40, args=(deepcopy(args),)))
    all_p.append(
        multiproc.Process(target=prepare_modelnet10, args=(deepcopy(args),)))
    all_p.append(
        multiproc.Process(target=prepare_shapenet, args=(deepcopy(args),)))
    all_p.append(
        multiproc.Process(target=download_modelnet_experiments_from_google_drive_and_unzip))
    all_p.append(
        multiproc.Process(target=download_shapenet_experiments_from_google_drive_and_unzip))
    run_all_processes(all_p)
    logger.info('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(sys.argv[0])

    parser.add_argument('--pts_mn40', type=int, default=1024,
                        help="number of points per modelNet40 object")
    parser.add_argument('--pts_mn10', type=int, default=1024,
                        help="number of points per modelNet10 object")
    parser.add_argument('--pts_shapenet', type=int, default=2048,
                        help="number of points per shapenet object")
    parser.add_argument('-md', '--mode', type=str, default="M",
                        help="mode used to compute graphs: M, P")
    parser.add_argument('-m', '--metric', type=str, default='euclidean',
                        help="metric for distance calculation (manhattan/euclidean)")
    parser.add_argument('--no-shuffle', dest='shuffle', action='store_false', default=True,
                        help="whether to shuffle data (1) or not (0) before saving")
    parser.add_argument('--regenerate',dest='regenerate',action='store_true',default=False,
                        help='regenerate from raw pointnet data or not (default: False)')

    args = parser.parse_args(sys.argv[1:])
    args.script_folder = os.path.dirname(os.path.abspath(__file__))

    main(args)
