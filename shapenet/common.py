# Copyright (C) 2018, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
from base import BaseExperiment
import caffecup
import caffecup.brew
import glog as logger
from kcnet import GlobalMaxPooling

###########################################################################

class BaseData(BaseExperiment):
    def data(self):
        total = self.train_data_shape()[0] * self.n_epoch()
        if total % self.train_batch_size() !=0:
            logger.warn('Training: training data size ({:d}*{:d}) is not divisible by batch size ({:d})!'.format(
                self.n_epoch(), self.train_data_shape()[0], self.train_batch_size()
            ))

        if self.test_data_shape()[0] % self.test_batch_size() !=0:
            logger.error('total number of testing data ({:d}) is not divisible by batch size ({:d})!'.format(
                self.test_data_shape()[0], self.test_batch_size()
            ))

        self.cc = caffecup.Designer(self.network_path)
        self.cc.name(self.network_name)
        self.cc.comment_bar('Data')
        self.train_data()
        self.test_data()

    def n_cls(self): return 50
    def n_category(self): return 16
    def n_epoch(self): return 400

    def outputs(self): raise NotImplementedError()

    def train_data_shape(self): raise NotImplementedError()
    def test_data_shape(self): raise NotImplementedError()

    def train_batch_size(self): raise NotImplementedError()
    def test_batch_size(self): raise NotImplementedError()

    def train_output_shapes(self): raise NotImplementedError()
    def test_output_shapes(self): raise NotImplementedError()

    def train_param_str(self): raise NotImplementedError()
    def test_param_str(self): raise NotImplementedError()

    def layer_name(self): raise NotImplementedError()

    def train_data(self):
        self.cc.pydata(
            self.outputs(),
            self.train_output_shapes(),
            module='io_layers',
            layer=self.layer_name(),
            param_str=self.train_param_str(),
            phase='TRAIN'
        )

    def test_data(self):
        self.cc.pydata(
            self.outputs(),
            self.test_output_shapes(),
            module='io_layers',
            layer=self.layer_name(),
            param_str=self.test_param_str(),
            phase='TEST',
            check_shape=True
        )

class BaseGraphDataSimple(BaseData):
    def layer_name(self): return 'IOShapeNetGraphLayer'
    def outputs(self): return ['X', 'Y', 'onehot', 'n_offset', 'G_indptr', 'G_indices']

    def train_output_shapes(self):
        S = self.train_data_shape()
        X_shape = [self.train_batch_size()*S[1],S[2]]
        Y_shape = [X_shape[0],]
        O_shape = [self.train_batch_size(), self.n_category()]
        N_shape = [self.train_batch_size(),]
        return [X_shape, Y_shape, O_shape, N_shape, [X_shape[0]+1,], ['nnz',]]

    def test_output_shapes(self):
        S = self.test_data_shape()
        X_shape = [self.test_batch_size()*S[1],S[2]]
        Y_shape = [X_shape[0],]
        O_shape = [self.test_batch_size(), self.n_category()]
        N_shape = [self.test_batch_size(),]
        return [X_shape, Y_shape, O_shape, N_shape, [X_shape[0]+1,], ['nnz',]]

    def global_pool_and_tile(self, i='X6', o='H_tile'):
        cc = self.cc
        N = self.train_data_shape()[1]

        GlobalMaxPooling(cc, i, 'n_offset', 'P') #BxK
        cc.concat(['P','onehot'],'H_pre',axis=-1) #Bx(K+16)
        Kh = cc.shape_of('H_pre')[-1]
        cc.reshape('H_pre','H',shape=[0,1,-1]) #Bx1x(K+16)
        cc.tile('H','H_tile_pre',axis=1,tiles=N).space() #BxNx(K+16)
        cc.reshape('H_tile_pre',o,shape=[-1,Kh]) #(B*N)x(K+16)

class BaseGraphDataFull(BaseGraphDataSimple):
    def layer_name(self): return 'IOShapeNetGraphLayer'
    def outputs(self): return ['X', 'Y', 'onehot', 'n_offset', 'G_indptr', 'G_indices', 'G_data']

    def train_output_shapes(self):
        S = self.train_data_shape()
        X_shape = [self.train_batch_size()*S[1],S[2]]
        Y_shape = [X_shape[0],]
        O_shape = [self.train_batch_size(), self.n_category()]
        N_shape = [self.train_batch_size(),]
        return [X_shape, Y_shape, O_shape, N_shape, [X_shape[0]+1,], ['nnz',], ['nnz',]]

    def test_output_shapes(self):
        S = self.test_data_shape()
        X_shape = [self.test_batch_size()*S[1],S[2]]
        Y_shape = [X_shape[0],]
        O_shape = [self.test_batch_size(), self.n_category()]
        N_shape = [self.test_batch_size(),]
        return [X_shape, Y_shape, O_shape, N_shape, [X_shape[0]+1,], ['nnz',], ['nnz',]]


class ShapeNet(BaseData):
    def layer_name(self): return 'IOShapeNetLayer'
    def outputs(self): return ['X', 'Y', 'onehot']

    def train_output_shapes(self):
        S = self.train_data_shape()
        X_shape = [self.train_batch_size(), S[1], S[2]]
        Y_shape = [X_shape[0], S[1]]
        O_shape = [X_shape[0], self.n_category()]
        return [X_shape, Y_shape, O_shape]
    def test_output_shapes(self):
        S = self.test_data_shape()
        X_shape = [self.test_batch_size(), S[1], S[2]]
        Y_shape = [X_shape[0], S[1]]
        O_shape = [X_shape[0], self.n_category()]
        return [X_shape, Y_shape, O_shape]

    def train_data_shape(self): return [14007, 2048, 3]
    def train_batch_size(self): return 64
    def test_data_shape(self): return [2874, 2048, 3]
    def test_batch_size(self): return 6
    def knn(self): raise NotImplementedError()

    def train_param_str(self):
        return "{'source':'data/shapenet_train_file_%dnn_GM.npy', 'batch_size':%d}" % (
                self.knn(), self.train_batch_size())

    def test_param_str(self):
        return "{'source':'data/shapenet_test_file_%dnn_GM.npy', 'batch_size':%d}" % (
                self.knn(), self.test_batch_size())

class ShapeNet9nn(ShapeNet):
    def knn(self): return 9

class ShapeNet18nn(ShapeNet):
    def knn(self): return 18

class ShapeNet36nn(ShapeNet):
    def knn(self): return 36

###########################################################################

class BaseClassifier(BaseData):
    def classifier(self):
        '''F -> loss, accuracy '''
        import caffecup.viz.draw_net as ccdraw
        cc = self.cc
        assert(isinstance(cc, caffecup.Designer))

        cc.comment_bar('Classifier')
        # fcreludrop = lambda i,o,n,d: cc.fc(i,o,n).relu(o).dropout(o,dropout_ratio=d).space()
        XKrelu = lambda i,o,n: cc.XK(i,o,n,axis=-1).relu(o).space()
        XKrelu('F' ,'F0',512)
        XKrelu('F0','F1',256)
        cc.fc( 'F1','Yp',self.n_cls(),axis=-1)

        cc.comment_bar('Final')
        cc.softmax_loss('Yp','Y',axis=-1)
        cc.accuracy('Yp','Y',axis=-1)

        cc.comment_blob_shape()
        cc.done(draw_net=ccdraw)
        print('written:'+cc.filepath)

class BaseClassifierWithDropout(BaseData):
    def classifier(self):
        '''F -> loss, accuracy '''
        import caffecup.viz.draw_net as ccdraw
        cc = self.cc
        assert(isinstance(cc, caffecup.Designer))

        cc.comment_bar('Classifier')
        XKreludrop = lambda i,o,n,d: cc.XK(i,o,n,axis=-1).relu(o).dropout(o,dropout_ratio=d).space()
        XKreludrop('F' ,'F0',512,0.3)
        XKreludrop('F0','F1',256,0.3)
        cc.fc( 'F1','Yp',self.n_cls(),axis=-1)

        cc.comment_bar('Final')
        cc.softmax_loss('Yp','Y',axis=-1)
        cc.accuracy('Yp','Y',axis=-1)

        cc.comment_blob_shape()
        cc.done(draw_net=ccdraw)
        print('written:'+cc.filepath)

class BaseClassifier2WithDropout(BaseData):
    def classifier(self):
        '''F -> loss, accuracy '''
        import caffecup.viz.draw_net as ccdraw
        cc = self.cc
        assert(isinstance(cc, caffecup.Designer))

        cc.comment_bar('Classifier')
        XKreludrop = lambda i,o,n,d: cc.XK(i,o,n,axis=-1).relu(o).dropout(o,dropout_ratio=d).space()
        XKreludrop('F' ,'F0',256,0.3)
        XKreludrop('F0','F1',256,0.3)
        XKreludrop('F1','F2',128,0.3)
        cc.fc( 'F2','Yp',self.n_cls(),axis=-1)

        cc.comment_bar('Final')
        cc.softmax_loss('Yp','Y',axis=-1)
        cc.accuracy('Yp','Y',axis=-1)

        cc.comment_blob_shape()
        cc.done(draw_net=ccdraw)
        print('written:'+cc.filepath)

###########################################################################

class Adam(BaseData):
    def solver(self):
        tts = caffecup.brew.TrainTestSolver(self.solver_path)
        cc = self.cc
        assert(isinstance(cc, caffecup.Designer))

        tts.build(
            traintest_net=cc.filepath,
            n_train_data=self.train_data_shape()[0],
            train_batch_size=self.train_batch_size(),
            n_test_data=self.test_data_shape()[0],
            test_batch_size=self.test_batch_size(),
            test_interval=self.n_epoch()*self.train_data_shape()[0]/self.train_batch_size()/100,
            test_initialization=True,
            n_epoch=self.n_epoch(),
            base_lr=0.001,
            solver_type='Adam',
            lr_policy='fixed',
            snapshot_folder=os.path.join(cc.filedir,'snapshot')
        )
        print('written:'+tts.filepath)
