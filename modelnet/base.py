# Copyright (C) 2018, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import sys

#setup path
SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
ROOT_FOLDER = os.path.abspath(os.path.join(SCRIPT_FOLDER,'..'))
PYCAFFE_FOLDER = os.path.abspath(os.path.join(ROOT_FOLDER,'../caffe/install/python'))
CAFFE_BIN = os.path.abspath(os.path.join(ROOT_FOLDER,'../caffe/install/bin/caffe'))
CAFFECUP_FOLDER = os.path.abspath(os.path.join(ROOT_FOLDER,'../caffecup'))
assert(os.path.exists(os.path.join(PYCAFFE_FOLDER,'caffe/_caffe.so')))
assert(os.path.exists(os.path.join(CAFFECUP_FOLDER,'caffecup/version.py')))
sys.path.insert(0, PYCAFFE_FOLDER)
sys.path.insert(1, CAFFECUP_FOLDER)
sys.path.insert(2, ROOT_FOLDER)
ADDITIONAL_PYTHONPATH_LIST = [ROOT_FOLDER, PYCAFFE_FOLDER]

CWD = os.getcwd()
DATA_FOLDER = os.path.join(CWD,'data')
if not os.path.exists(DATA_FOLDER):
    raise RuntimeError('Please make sure your data folder exists: '+DATA_FOLDER)

__all__=[
    'BaseExperiment',
    'ROOT_FOLDER',
    'PYCAFFE_FOLDER',
    'CAFFECUP_FOLDER',
    'ADDITIONAL_PYTHONPATH_LIST',
    'CWD',
    'DATA_FOLDER'
]

class BaseExperiment(object):
    def __init__(self, network_name, network_path=None):
        self.network_name = network_name
        if network_path is None:
            network_path = './experiments/{}/{}.ptt'.format(network_name, network_name)
        self.network_path = network_path
        self.solver_path = os.path.join(os.path.dirname(self.network_path),'solve.ptt')
        self.cc = None #should be setup by child class

    def data(self):
        raise NotImplementedError()

    def network(self):
        raise NotImplementedError()

    def solver(self):
        raise NotImplementedError()

    def design(self):
        self.data()
        self.network()
        self.solver()
        import inspect
        print('Next:\n'
              ' 1. check generated files\n'
              ' 2. start training (append --dryrun to check; append -h for more help):\n'
              'python %s brew' % (inspect.getmodule(inspect.stack()[2][0]).__file__))

    def brew(self):
        import caffecup.brew.run_solver as run_solver
        args = run_solver.get_args(sys.argv)
        args.caffe = CAFFE_BIN
        args.additional_pythonpath += ADDITIONAL_PYTHONPATH_LIST
        try:
            run_solver.main(args)
        except IOError as e:
            import inspect
            print(e)
            print('Did you forget to execute the following command before brew?\n\tpython %s' % (
                inspect.getmodule(inspect.stack()[2][0]).__file__))

    def clean(self):
        from caffecup.viz.plot_traintest_log import get_testing_accuracy
        from caffecup.brew.run_solver import solver2dict
        import numpy as np
        sdict = solver2dict(self.solver_path)
        snapshot_prefix = sdict['snapshot_prefix']
        logfile = 'logs/{}.log'.format(os.path.splitext(os.path.basename(snapshot_prefix))[0])
        print('parsing log: '+logfile)
        assert(os.path.exists(logfile))
        accu_iters, accuracies = get_testing_accuracy(open(logfile,'r').read())
        if accuracies is None:
            print('exit, because log is not loaded!')
            return
        max_accu_pos = np.argmax(accuracies)
        max_accu_iter = accu_iters[max_accu_pos]
        max_accu = accuracies[max_accu_pos]
        keep_file = '{}_iter_{}.caffemodel'.format(snapshot_prefix, max_accu_iter)
        print('Max testing accuracy (iter={}): {}'.format(max_accu_iter, max_accu))
        print('To keep only: '+keep_file)
        assert(os.path.exists(keep_file))
        finished_pattern = '{}_iter_{}.caffemodel'.format(snapshot_prefix, sdict['max_iter'])
        if not os.path.exists(finished_pattern):
            print('exit, because training not finished yet!')
            return
        snapshot_folder = os.path.dirname(snapshot_prefix)
        all_files = sorted(os.listdir(snapshot_folder))
        is_dryrun = (len(sys.argv)>2 and sys.argv[2]=='--dryrun')
        for fi in all_files:
            fi = os.path.join(snapshot_folder,fi)
            if fi!=keep_file:
                if is_dryrun:
                    print('clean: '+fi)
                else:
                    os.remove(fi)
            else:
                print('KEEP: '+fi)

    @staticmethod
    def plot():
        import caffecup.viz.plot_traintest_log as plotter
        try:
            args = plotter.get_args(sys.argv)
        except ValueError:
            sys.argv.append(os.path.join(CWD,'logs'))
            args = plotter.get_args(sys.argv)

        args.all_in_one = 1
        if args.axis_left is None:
            args.axis_left = [0,0.6]
        if args.axis_right is None:
            args.axis_right = [0.8,0.95]
        plotter.main(args)

    def main(self):
        if len(sys.argv[1:])==0:
            self.design()
        else:
            if sys.argv[1]=='brew':
                sys.argv.pop(1)
                sys.argv.insert(1, '-s')
                sys.argv.insert(2, self.solver_path)
                self.brew()
            elif sys.argv[1]=='clean':
                self.clean()
            else:
                raise NotImplementedError()

if __name__ == '__main__':
    BaseExperiment.plot() #plot all logs
