# Copyright (C) 2018, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
from base import BaseExperiment
import caffecup
import caffecup.viz.draw_net as ccdraw
import caffecup.brew
import glog as logger

###########################################################################

class BaseData(BaseExperiment):
    def data(self):
        total = self.train_data_shape()[0] * self.n_epoch()
        if total % self.train_batch_size() !=0:
            logger.warn('Training: training data size ({:d}*{:d}) is not divisible by batch size ({:d})!'.format(
                self.n_epoch(), self.train_data_shape()[0], self.train_batch_size()
            ))

        if self.test_data_shape()[0] % self.test_batch_size() !=0:
            logger.warn('total number of testing data ({:d}) is not divisible by batch size ({:d})!'.format(
                self.test_data_shape()[0], self.test_batch_size()
            ))

        self.cc = caffecup.Designer(self.network_path)
        self.cc.name(self.network_name)
        self.cc.comment_bar('Data')
        self.train_data()
        self.test_data()

    def n_cls(self): raise NotImplementedError()
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

class BaseGraphData(BaseData):
    def layer_name(self): return 'IOModelNetGraphLayer'

class GraphDataSimple(BaseGraphData):
    def outputs(self): return ['X', 'Y', 'n_offset', 'G_indptr', 'G_indices']

    def train_output_shapes(self):
        S = self.train_data_shape()
        X_shape = [self.train_batch_size()*S[1],S[2]]
        Y_shape = [self.train_batch_size(),]
        return [X_shape, Y_shape, Y_shape, [X_shape[0]+1,], ['nnz',]]

    def test_output_shapes(self):
        S = self.test_data_shape()
        X_shape = [self.test_batch_size()*S[1],S[2]]
        Y_shape = [self.test_batch_size(),]
        return [X_shape, Y_shape, Y_shape, [X_shape[0]+1,], ['nnz',]]

class GraphDataFull(GraphDataSimple):
    def outputs(self): return ['X', 'Y', 'n_offset', 'G_indptr', 'G_indices', 'G_data']

    def train_output_shapes(self):
        S = self.train_data_shape()
        X_shape = [self.train_batch_size()*S[1],S[2]]
        Y_shape = [self.train_batch_size(),]
        return [X_shape, Y_shape, Y_shape, [X_shape[0]+1,], ['nnz',], ['nnz',]]

    def test_output_shapes(self):
        S = self.test_data_shape()
        X_shape = [self.test_batch_size()*S[1],S[2]]
        Y_shape = [self.test_batch_size(),]
        return [X_shape, Y_shape, Y_shape, [X_shape[0]+1,], ['nnz',], ['nnz',]]


class ModelNet40(BaseData):
    def train_data_shape(self): return [9840,1024,3]
    def train_batch_size(self): return 64
    def test_data_shape(self): return [2468,1024,3]
    def test_batch_size(self): return 617
    def n_cls(self): return 40
    def outputs(self): return ['X','Y']
    def layer_name(self): return 'IOModelNetLayer'

    def train_output_shapes(self):
        X_shape = self.train_data_shape()
        X_shape[0] = self.train_batch_size()
        Y_shape = [X_shape[0],]
        return [X_shape, Y_shape]
    def train_param_str(self):
        return "{'source':'data/modelNet40_train_file_16nn_GM.npy', 'batch_size':%d}" % (
                self.train_batch_size())

    def test_output_shapes(self):
        X_shape = self.test_data_shape()
        X_shape[0] = self.test_batch_size()
        Y_shape = [X_shape[0],]
        return [X_shape, Y_shape]
    def test_param_str(self):
        return "{'source':'data/modelNet40_test_file_16nn_GM.npy', 'batch_size':%d}" % (
                self.test_batch_size())

class ModelNet40orthoRonline(ModelNet40):
    def train_param_str(self):
        return "{'source':'data/modelNet40_train_file_16nn_GM.npy',"\
               " 'batch_size':%d, 'rand_rotation':'ortho', 'rand_online':True, 'random_seed':3827289}" % (
                self.train_batch_size())
class ModelNet40GraphorthoRonline(ModelNet40):
    def train_param_str(self):
        return "{'source':'data/modelNet40_train_file_16nn_GM.npy',"\
               " 'batch_size':%d, 'mode':'M', 'rand_rotation':'ortho', 'rand_online':True, 'random_seed':3827289}" % (
                self.train_batch_size())

class ModelNet40GraphSimple(GraphDataSimple, ModelNet40):
    def test_batch_size(self): return 4

    def train_param_str(self):
        return "{'source':'data/modelNet40_train_file_16nn_GM.npy', 'batch_size':%d, 'mode':'M'}" % (
                self.train_batch_size())
    def test_param_str(self):
        return "{'source':'data/modelNet40_test_file_16nn_GM.npy', 'batch_size':%d, 'mode':'M'}" % (
                self.test_batch_size())

class ModelNet40GraphFull(ModelNet40GraphSimple, GraphDataFull):
    pass


class ModelNet10(ModelNet40):
    def train_data_shape(self): return [3991,1024,3]
    def train_batch_size(self): return 65
    def test_data_shape(self): return [908,1024,3]
    def test_batch_size(self): return 908
    def n_cls(self): return 10
    def train_param_str(self):
        return "{'source':'data/modelNet10_train.npy', 'batch_size':%d}" % (
                self.train_batch_size())
    def test_param_str(self):
        return "{'source':'data/modelNet10_test.npy', 'batch_size':%d}" % (
                self.test_batch_size())

class ModelNet10GraphSimple(GraphDataSimple, ModelNet10):
    def test_batch_size(self): return 4

    def train_param_str(self):
        return "{'source':'data/modelNet10_train_16nn_GM.npy', 'batch_size':%d, 'mode':'M'}" % (
                self.train_batch_size())
    def test_param_str(self):
        return "{'source':'data/modelNet10_test_16nn_GM.npy', 'batch_size':%d, 'mode':'M'}" % (
                self.test_batch_size())

class ModelNet10GraphFull(ModelNet10GraphSimple, GraphDataFull):
    pass


###########################################################################

class BaseClassifier(BaseData):
    def classifier(self):
        '''F -> loss, accuracy '''
        cc = self.cc
        assert(isinstance(cc, caffecup.Designer))

        cc.comment_bar('Classifier')
        fcreludrop = lambda i,o,n,d: cc.fc(i,o,n).relu(o).dropout(o,dropout_ratio=d).space()
        fcreludrop('F','F0',512,0.3)
        fcreludrop('F0','F1',256,0.3)
        cc.fc('F1','Yp',self.n_cls())

        cc.comment_bar('Final')
        cc.softmax_loss('Yp','Y')
        cc.accuracy('Yp','Y')

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
            test_interval=self.n_epoch()*self.train_data_shape()[0]/self.train_batch_size()/50,
            test_initialization=True,
            n_epoch=self.n_epoch(),
            base_lr=0.001,
            solver_type='Adam',
            lr_policy='fixed',
            snapshot_folder=os.path.join(cc.filedir,'snapshot')
        )
        print('written:'+tts.filepath)
