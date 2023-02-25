# Copyright (C) 2018, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import sys
import numpy as np
import scipy.sparse as sparse
import caffe
from utils import rand_rotation_matrix, rand_ortho_rotation_matrix


def fastprint(str):
    print(str)
    sys.stdout.flush()


class BaseIOLayer(caffe.Layer):
    ''' base io class supporting batch, random rotation and noisy points augmentation

    params = {
    'source':'data/modelNet_train.npy',
    'batch_size': 64
    'random_seed': -1 #seed for random replace some portion with noise points
    'num_noise_pts': 0
    'rand_rotation': '' # ''|'ortho'|'full'
    'rand_online': False,
    }
    '''
    def set_member(self, name, default=None):
        setattr(self, name,
                self.params[name]
                if default is None or self.params.has_key(name)
                else
                default)

    def get_batch_indices(self, start_id):
        return [i%self.n_all_data for i in xrange(start_id, start_id+self.batch_size)]

    def next_batch(self):
        ''' invoked in self.forward(bottom,top) to move to next batch '''
        self.ith = (self.ith + self.batch_size) % self.n_all_data

    def augment_data(self, xyz):
        ''' invoked either online in self.reshape(bottom,top) or offline in self.setup(bottom,top)

        xyz <BxNxD>: batched point clouds
        '''
        if self.num_noise_pts>0: #add noise online
            assert(self.num_noise_pts<xyz.shape[1])
            for kth in xrange(xyz.shape[0]):
                ids = np.random.randint(0, xyz.shape[1],
                                        size=(self.num_noise_pts,),
                                        dtype=int)
                xyz[kth][ids] = np.random.rand(
                    self.num_noise_pts, xyz.shape[2]) * 2 - 1.0

        if self.rand_rotation: #make random rotation
            rand_rot_func = rand_ortho_rotation_matrix if self.rand_rotation.lower()=='ortho' else rand_rotation_matrix
            for kth in xrange(xyz.shape[0]):
                M = rand_rot_func()
                xyz[kth, ...]=np.dot(xyz[kth], M)

    def setup(self, bottom, top):
        self.params = eval(self.param_str)
        # fastprint(self.params)

        self.set_member('source')
        assert(os.path.exists(self.source))

        self.set_member('num_noise_pts',0)
        self.set_member('rand_rotation','')
        self.set_member('random_seed',-1)
        self.set_member('rand_online', False)

        if self.random_seed>0:
            np.random.seed(np.uint32(self.random_seed))
        if self.num_noise_pts>0 or self.rand_rotation:
            assert(self.random_seed>0)

        self.set_member('batch_size')
        assert(self.batch_size>0)

        self.load_data()
        assert(hasattr(self, 'all_data'))
        assert(hasattr(self, 'all_label'))

        if not self.rand_online:
            self.augment_data(self.all_data)

        self.ith = 0
        self.old_ith = -1
        self.n_all_data = self.all_data.shape[0]
        assert(self.batch_size<=self.n_all_data)
        assert(self.n_all_data==len(self.all_label))

        # fastprint('n_all_data={}'.format(self.n_all_data))

    def reshape(self, bottom, top):
        if self.ith==self.old_ith:
            return
        if self.ith<self.old_ith: #restarted
            new_idx = np.random.permutation(self.n_all_data)
            self.reshuffle(new_idx)
        self.old_ith = self.ith

        bids = self.get_batch_indices(self.ith)
        self.update_current_data(bids, top)

    def backward(self, top, propagate_down, bottom): pass

    def forward(self, bottom, top): raise NotImplementedError()

    def load_data(self):
        ''' invoked only once in self.setup(bottom,top) '''
        raise NotImplementedError()
    def reshuffle(self, new_idx):
        ''' invoked in self.reshape(bottom,top) after each epoch '''
        raise NotImplementedError()
    def update_current_data(self, bids, top):
        ''' invoked in self.reshape(bottom,top) to load current batch with bids '''
        raise NotImplementedError()


class IOModelNetLayer(BaseIOLayer):
    '''
    IOModelNetLayer -> X, Y

    X <BxNx3>:   batched 3D Points, each containing N points representing a object instance
    Y <B>:       class label for each object instance in the batch

    params = {
    'source':'data/modelNet_train.npy',
    'batch_size': 64
    'random_seed': -1 #seed for random replace some portion with noise points
    'num_noise_pts': 0
    'rand_rotation': '' # ''|'ortho'|'full'
    'rand_online': False,
    }
    '''

    def load_data(self):
        raw_data = np.load(self.source).item()
        assert(isinstance(raw_data, dict))
        self.all_data = raw_data['data'] #[BxNx3]
        self.all_label = raw_data['label']
        assert(len(self.all_data.shape)==3)
        assert(self.all_data.shape[0]==self.all_label.shape[0])
        assert(self.all_data.shape[-1]==3)

    def reshuffle(self, new_idx):
        self.all_data = self.all_data[new_idx]
        self.all_label = self.all_label[new_idx]

    def update_current_data(self, bids, top):
        if self.rand_online:
            self.xyz=np.array(self.all_data[bids])
            self.augment_data(self.xyz)
        else:
            self.xyz=self.all_data[bids]

        self.lb=self.all_label[bids]

        assert(self.xyz.shape[0]==self.batch_size)
        assert(len(top)==2)
        top[0].reshape(*self.xyz.shape)
        top[1].reshape(*self.lb.shape)

    def forward(self, bottom, top):
        top[0].data[...] = self.xyz
        top[1].data[...] = self.lb
        self.next_batch()


class IOModelNetGraphLayer(BaseIOLayer):
    '''
    IOModelNetGraphLayer -> X, Y, n_offset, G_indptr, G_indices[, G_data]

    X <(B*N)x3>: batched 3D Points, each containing N points representing a object instance
    Y <B,>: class label for each object instance in the batch
    n_offset <B,>: end indices for each data instance in the batch
    G_indptr, G_indices, G_data: batched sparse G matrix (block diagonal), G_data might be omitted

    params = {
    'source':'data/modelNet_train.npy',
    'batch_size': 64
    'mode': 'P' # 'M'
    'random_seed': -1 #seed for random replace some portion with noise points
    'num_noise_pts': 0
    'rand_rotation': '' # ''|'ortho'|'full'
    'rand_online': False,
    }
    '''

    def load_data(self):
        self.set_member('mode')
        assert(self.mode in ['M','P'])

        #Graph data should be recomputed if X changes, thus num_noise_pts should always be 0
        assert(self.num_noise_pts==0)
        raw_data = np.load(self.source).item()
        assert(isinstance(raw_data, dict))
        self.all_data = raw_data['data'] #[BxNx3]
        self.all_label = raw_data['label']
        raw_graph = raw_data['graph']
        self.all_graph = np.asarray([g[self.mode] for g in raw_graph])
        assert(len(self.all_data.shape)==3)
        assert(self.all_data.shape[0]==self.all_label.shape[0])
        assert(self.all_data.shape[-1]==3)
        assert(len(self.all_graph)==self.all_data.shape[0])

    def reshuffle(self, new_idx):
        self.all_data = self.all_data[new_idx]
        self.all_label = self.all_label[new_idx]
        self.all_graph = self.all_graph[new_idx]

    def update_current_data(self, bids, top):
        if self.rand_online:
            self.xyz=np.array(self.all_data[bids])
            self.augment_data(self.xyz)
        else:
            self.xyz=self.all_data[bids]

        self.lb=self.all_label[bids]

        self.xyz = self.xyz.reshape((-1,self.xyz.shape[-1])) #reshape to (B*N)x3
        self.n_offset = (np.arange(self.lb.shape[0])+1)*self.all_data.shape[1]
        self.G = sparse.block_diag(self.all_graph[bids], format='csr')

        assert(self.xyz.shape[0]<16777216) #check graph_pooling_layer Reshape() for more information
        assert(self.G.indices.shape[0]<16777216) #if assertion failed, use a smaller self.batch_size!!
        assert(isinstance(self.G, sparse.csr_matrix))

        assert(self.xyz.shape[0]==self.batch_size*self.all_data.shape[1])
        assert(len(top) in [5,6])
        top[0].reshape(*self.xyz.shape)
        top[1].reshape(*self.lb.shape)
        top[2].reshape(*self.n_offset.shape)
        top[3].reshape(*self.G.indptr.shape)
        top[4].reshape(*self.G.indices.shape)
        if len(top)>5:
            top[5].reshape(*self.G.data.shape)

    def forward(self, bottom, top):
        top[0].data[...] = self.xyz
        top[1].data[...] = self.lb
        top[2].data[...] = self.n_offset
        top[3].data[...] = self.G.indptr
        top[4].data[...] = self.G.indices
        if len(top)>5:
            top[5].data[...] = self.G.data
        self.next_batch()


class IOShapeNetLayer(BaseIOLayer):
    '''
    IOShapeNetLayer -> X, Y, category
    '''

    def load_data(self):
        raw_data = np.load(self.source).item()
        assert(isinstance(raw_data, dict))
        self.all_data = raw_data['data'] #[BxNx3]
        self.all_seglabel = raw_data['seg_label']
        self.all_label = raw_data['label'] #[B,], category information

        assert(len(self.all_data.shape)==3)
        assert(self.all_data.shape[0]==self.all_label.shape[0])
        assert(self.all_data.shape[-1]==3)
        assert(len(self.all_seglabel.shape)==2)
        assert(self.all_seglabel.shape[:2]==self.all_data.shape[:2])

    def reshuffle(self, new_idx):
        self.all_data = self.all_data[new_idx]
        self.all_seglabel = self.all_seglabel[new_idx]
        self.all_label = self.all_label[new_idx]

    def update_current_data(self, bids, top):
        if self.rand_online:
            self.xyz=np.array(self.all_data[bids])
            self.augment_data(self.xyz)
        else:
            self.xyz=self.all_data[bids] #BxNx3

        self.n_shape_category = 16

        self.seglb=self.all_seglabel[bids] #BxN
        self.lb=self.all_label[bids] #B
        #one-hot encoding of shape category
        self.onehot = np.zeros((self.batch_size, self.n_shape_category), dtype=np.float32) #Bx16
        self.onehot[np.arange(len(self.lb)), self.lb] = 1.

        assert(self.xyz.shape[0]==self.batch_size)
        assert(len(top)==3)
        top[0].reshape(*self.xyz.shape)
        top[1].reshape(*self.seglb.shape)
        top[2].reshape(*self.onehot.shape)

    def forward(self, bottom, top):
        top[0].data[...] = self.xyz
        top[1].data[...] = self.seglb
        top[2].data[...] = self.onehot
        self.next_batch()


class IOShapeNetGraphLayer(BaseIOLayer):
    '''
    IOShapeNetLayer -> X, Y, category, n_offset, G_indptr, G_indices[, G_data]
    '''

    def load_data(self):
        self.mode = 'M'

        #Graph data should be recomputed if X changes, thus num_noise_pts should always be 0
        assert(self.num_noise_pts==0)
        raw_data = np.load(self.source).item()
        assert(isinstance(raw_data, dict))
        self.all_data = raw_data['data'] #[BxNx3]
        self.all_seglabel = raw_data['seg_label']
        self.all_label = raw_data['label'] #[B,], category information

        raw_graph = raw_data['graph']
        self.all_graph = np.asarray([g[self.mode] for g in raw_graph])
        assert(len(self.all_data.shape)==3)
        assert(self.all_data.shape[0]==self.all_label.shape[0])
        assert(self.all_data.shape[-1]==3)
        assert(len(self.all_graph)==self.all_data.shape[0])

    def reshuffle(self, new_idx):
        self.all_data = self.all_data[new_idx]
        self.all_seglabel = self.all_seglabel[new_idx]
        self.all_label = self.all_label[new_idx]
        self.all_graph = self.all_graph[new_idx]

    def update_current_data(self, bids, top):
        if self.rand_online:
            self.xyz=np.array(self.all_data[bids])
            self.augment_data(self.xyz)
        else:
            self.xyz=self.all_data[bids] #BxNx3

        self.n_shape_category = 16

        self.seglb=self.all_seglabel[bids] #BxN
        self.lb=self.all_label[bids] #B
        #one-hot encoding of shape category
        self.onehot = np.zeros((self.batch_size, self.n_shape_category), dtype=np.float32) #Bx16
        self.onehot[np.arange(self.batch_size), self.lb] = 1.

        self.xyz = self.xyz.reshape((-1,self.xyz.shape[-1])) #reshape to (B*N)x3
        self.seglb = self.seglb.reshape(-1) #reshape to (B*N,)
        self.n_offset = (np.arange(self.lb.shape[0])+1)*self.all_data.shape[1] #(B,)
        self.G = sparse.block_diag(self.all_graph[bids], format='csr')

        assert(self.xyz.shape[0]<16777216) #check graph_pooling_layer Reshape() for more information
        assert(self.G.indices.shape[0]<16777216) #if assertion failed, use a smaller self.batch_size!!
        assert(isinstance(self.G, sparse.csr_matrix))

        assert(self.xyz.shape[0]==self.batch_size*self.all_data.shape[1])
        assert(len(top) in [6,7])
        top[0].reshape(*self.xyz.shape)
        top[1].reshape(*self.seglb.shape)
        top[2].reshape(*self.onehot.shape)
        top[3].reshape(*self.n_offset.shape)
        top[4].reshape(*self.G.indptr.shape)
        top[5].reshape(*self.G.indices.shape)
        if len(top)>6:
            top[6].reshape(*self.G.data.shape)

    def forward(self, bottom, top):
        top[0].data[...] = self.xyz
        top[1].data[...] = self.seglb
        top[2].data[...] = self.onehot
        top[3].data[...] = self.n_offset
        top[4].data[...] = self.G.indptr
        top[5].data[...] = self.G.indices
        if len(top)>6:
            top[6].data[...] = self.G.data
        self.next_batch()
