# Copyright (C) 2018, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import matplotlib as mpl
mpl.use('Agg')
import os
import sys
import time
import argparse
import glog as logger
import numpy as np

import base
from caffecup.brew.run_solver import solver2dict, fastprint, create_if_not_exist
import caffe
from visualize import kernel2img, pts2img, cmap

#http://stackoverflow.com/a/4836734/2303236
def natural_sort(l):
    import re
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

class ShapeNetTester(object):

    def __init__(self):
        self.seg_classes = {
            'Airplane':     [0, 1, 2, 3],
            'Bag':          [4, 5],
            'Cap':          [6, 7],
            'Car':          [8, 9, 10, 11],
            'Chair':        [12, 13, 14, 15],
            'Earphone':     [16, 17, 18],
            'Guitar':       [19, 20, 21],
            'Knife':        [22, 23],
            'Lamp':         [24, 25, 26, 27],
            'Laptop':       [28, 29],
            'Motorbike':    [30, 31, 32, 33, 34, 35],
            'Mug':          [36, 37],
            'Pistol':       [38, 39, 40],
            'Rocket':       [41, 42, 43],
            'Skateboard':   [44, 45, 46],
            'Table':        [47, 48, 49]
        }

        self.part_colors = [
            [0.65, 0.95, 0.05], [0.35, 0.05, 0.35], [0.65, 0.35, 0.65], [0.95, 0.95, 0.65], [0.95, 0.65, 0.05], [0.35, 0.05, 0.05], [0.65, 0.05, 0.05], [0.65, 0.35, 0.95], [0.05, 0.05, 0.65], [0.65, 0.05, 0.35], [0.05, 0.35, 0.35], [0.65, 0.65, 0.35], [0.35, 0.95, 0.05], [0.05, 0.35, 0.65], [0.95, 0.95, 0.35], [0.65, 0.65, 0.65], [0.95, 0.95, 0.05], [0.65, 0.35, 0.05], [0.35, 0.65, 0.05], [0.95, 0.65, 0.95], [0.95, 0.35, 0.65], [0.05, 0.65, 0.95], [0.65, 0.95, 0.65], [0.95, 0.35, 0.95], [0.05, 0.05, 0.95], [0.65, 0.05, 0.95], [0.65, 0.05, 0.65], [0.35, 0.35, 0.95], [0.95, 0.95, 0.95], [0.05, 0.05, 0.05], [0.05, 0.35, 0.95], [0.65, 0.95, 0.95], [0.95, 0.05, 0.05], [0.35, 0.95, 0.35], [0.05, 0.35, 0.05], [0.05, 0.65, 0.35], [0.05, 0.95, 0.05], [0.95, 0.65, 0.65], [0.35, 0.95, 0.95], [0.05, 0.95, 0.35], [0.95, 0.35, 0.05], [0.65, 0.35, 0.35], [0.35, 0.95, 0.65], [0.35, 0.35, 0.65], [0.65, 0.95, 0.35], [0.05, 0.95, 0.65], [0.65, 0.65, 0.95], [0.35, 0.05, 0.95], [0.35, 0.65, 0.95], [0.35, 0.05, 0.65]
        ]

        self.seg_label_to_cat = {} # {0:Airplane, 1:Airplane, ...49:Table}
        for cat in self.seg_classes.keys():
            for label in self.seg_classes[cat]:
                self.seg_label_to_cat[label] = cat

        caffe.set_mode_gpu()
        caffe.set_device(0)

    def visualize_core(self, traintest, weight_path, test_iters, vis_KC, vis_pred, vis_gt):
        net = caffe.Net(traintest, weights=weight_path, phase=caffe.TEST)

        VIS_DIR = os.path.join(
            base.ROOT_FOLDER,'visualization','shapenet',
            os.path.splitext(os.path.basename(weight_path))[0]
        )
        create_if_not_exist(VIS_DIR)

        ## visualize kernel
        kernels = np.array(net.params['KC'][0].data, dtype=np.float32) #<LxMx3>
        kn_min, kn_max = -0.5, 0.5
        for k in xrange(kernels.shape[0]):
            fname = os.path.join(VIS_DIR, "kernel_{:02d}.png".format(k))
            kernel2img(
                kernels[k],kn_min,kn_max
            ).save(fname)
            fastprint(fname)

        ## visualize kernel responses and segmentation results
        for ti in range(test_iters):
            net.forward()

            xyz = np.squeeze(np.array(net.blobs['X'].data)) #<BxNx3> | <B*N,3>
            true_label = np.array(net.blobs['Y'].data, dtype=np.int32) #<BxN> | <B*N,>, ground truth seg_label
            assert(len(true_label.shape) in [1,2])
            pred_logits = np.array(net.blobs['Yp'].data) #<BxNx50> | <B*N,50>
            one_hot = np.array(net.blobs['onehot'].data) #<Bx16>
            kernel_responses = np.array(net.blobs['Xc'].data) #<B*N,L>
            batch_size = one_hot.shape[0]
            n_cls = pred_logits.shape[-1]
            n_pts = true_label.shape[-1]

            if len(true_label.shape)==1: #<B*N,>
                n_pts = true_label.shape[0]/batch_size
                xyz = xyz.reshape(batch_size,n_pts,-1) #<BxNx3>
                kernel_responses = kernel_responses.reshape(batch_size,n_pts,-1) #<BxNxL>
                true_label = true_label.reshape((batch_size,n_pts)) #<BxN>
                pred_logits = pred_logits.reshape((batch_size,n_pts,n_cls)) #<BxNx50>
            else: #<BxN>
                assert(true_label.shape[0]==batch_size)
            assert(true_label.shape==(batch_size,n_pts))
            assert(pred_logits.shape==(batch_size,n_pts,n_cls))

            # calc pred_label
            pred_label = np.zeros((batch_size, n_pts),dtype=np.int32) #<BxN>
            for bi in range(batch_size):
                cat = self.seg_label_to_cat[true_label[bi,0]] #string
                logits = pred_logits[bi][:,self.seg_classes[cat]] #<NxK>
                # Constrain pred_label to the groundtruth classes (selected by seg_classes[cat])
                pred_label[bi, :] = np.argmax(logits, 1) + self.seg_classes[cat][0]

            # vis
            for bi in range(batch_size):
                cat = self.seg_label_to_cat[true_label[bi,0]] #string
                the_folder = os.path.join(VIS_DIR, cat)
                create_if_not_exist(the_folder)
                fname_prefix = os.path.join(the_folder, '{:04d}'.format(ti*batch_size+bi))
                pts2img(
                    xyz[bi],clr=None,cmap='binary'
                ).save(fname_prefix+'_orig.png')
                if vis_KC:
                    for k in range(kernel_responses.shape[-1]):
                        fname = fname_prefix+'_kc{:02d}.png'.format(k)
                        pts2img(
                            xyz[bi],
                            kernel_responses[bi,:,k],
                            cmap
                        ).save(fname)
                        fastprint(fname)
                if vis_pred:
                    fname = fname_prefix+'_pred.png'
                    pts2img(
                        xyz[bi],
                        clr=np.array([self.part_colors[l] for l in pred_label[bi]]),
                        cmap=None
                    ).save(fname)
                    fastprint(fname)
                if vis_gt:
                    fname = fname_prefix+'_gt.png'
                    pts2img(
                        xyz[bi],
                        clr=np.array([self.part_colors[l] for l in true_label[bi]]),
                        cmap=None
                    ).save(fname)
                    fastprint(fname)

    def visualize(self, solver_path, args):
        sdict = solver2dict(solver_path)
        network_path = sdict['net']
        test_iter = int(sdict['test_iter'])
        weight_path = args.weight

        self.visualize_core(network_path, weight_path, test_iter, True, True, True)

    def evaluate_core(self, traintest, weight_path, test_iters):
        shape_ious = {cat:[] for cat in self.seg_classes.keys()}

        net = caffe.Net(traintest, weights=weight_path, phase=caffe.TEST)

        for ti in range(test_iters):
            net.forward()

            true_label = np.array(net.blobs['Y'].data, dtype=np.int32) #<BxN> | <B*N,>
            assert(len(true_label.shape) in [1,2])
            pred_logits = np.array(net.blobs['Yp'].data) #<BxNx50> | <B*N,50>
            one_hot = np.array(net.blobs['onehot'].data) #<Bx16>
            batch_size = one_hot.shape[0]
            n_cls = pred_logits.shape[-1]
            n_pts = true_label.shape[-1]
            if len(true_label.shape)==1: #<B*N,>
                n_pts = true_label.shape[0]/batch_size
                true_label = true_label.reshape((batch_size,n_pts))
                pred_logits = pred_logits.reshape((batch_size,n_pts,n_cls))
            else: #<BxN>
                assert(true_label.shape[0]==batch_size)
            assert(true_label.shape==(batch_size,n_pts))
            assert(pred_logits.shape==(batch_size,n_pts,n_cls))

            # calc pred_label
            pred_label = np.zeros((batch_size, n_pts),dtype=np.int32) #<BxN>
            for bi in range(batch_size):
                cat = self.seg_label_to_cat[true_label[bi,0]] #string
                logits = pred_logits[bi][:,self.seg_classes[cat]] #<NxK>
                # Constrain pred_label to the groundtruth classes (selected by seg_classes[cat])
                pred_label[bi, :] = np.argmax(logits, 1) + self.seg_classes[cat][0]

            for bi in range(batch_size):
                segp = pred_label[bi, :]
                segl = true_label[bi, :]
                cat = self.seg_label_to_cat[segl[0]]
                part_ious = [0.0 for _ in range(len(self.seg_classes[cat]))]
                for l in self.seg_classes[cat]:
                    if (np.sum(segl == l) == 0) and (np.sum(segp == l) == 0):  # part is not present, no prediction as well
                        iou = 1.0
                    else:
                        iou = np.sum((segl == l) & (segp == l)) / float(np.sum((segl == l) | (segp == l)))
                    part_ious[l - self.seg_classes[cat][0]] = iou
                shape_ious[cat].append(np.mean(part_ious))

        all_shape_ious = []
        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:
                all_shape_ious.append(iou)
            shape_ious[cat] = np.mean(shape_ious[cat])

        cls_miou = np.mean(shape_ious.values())
        ins_miou = np.mean(all_shape_ious)

        ret = dict(shape_ious)
        ret['cls'] = cls_miou
        ret['ins'] = ins_miou
        return ret

    def evaluate(self, solver_path, clean=False):
        sdict = solver2dict(solver_path)
        snapshot_prefix = sdict['snapshot_prefix']
        network_path = sdict['net']
        test_iter = int(sdict['test_iter'])
        snapshot_folder = os.path.dirname(snapshot_prefix)
        all_files = natural_sort(os.listdir(snapshot_folder))
        all_dict = []
        max_cls_miou, max_ins_miou = -1, -1
        max_cls_at, max_ins_at = -1, -1
        for fi in all_files:
            if fi.endswith('.caffemodel'):
                fi = os.path.join(snapshot_folder,fi)
                ith_iter = int(fi.replace(snapshot_prefix,'').replace('_iter_','').replace('.caffemodel',''))
                ret = self.evaluate_core(network_path, fi, test_iter)
                all_dict.append((ith_iter, ret))
                fastprint('%6d: %s' % (ith_iter, ' '.join(['%s=%.4f'%(k,ret[k]) for k in sorted(ret.keys())])))
                if max_ins_miou<ret['ins']:
                    max_ins_at = ith_iter
                if max_cls_miou<ret['cls']:
                    max_cls_at = ith_iter
                max_cls_miou = max(max_cls_miou, ret['cls'])
                max_ins_miou = max(max_ins_miou, ret['ins'])
        fastprint('MAX mIoU:')
        fastprint('cls(@%6d)=%.4f ins(@%6d)=%.4f' % (max_cls_at, max_cls_miou, max_ins_at, max_ins_miou))

        if clean:
            for fi in all_files:
                if fi.endswith('.caffemodel'):
                    fi = os.path.join(snapshot_folder,fi)
                    ith_iter = int(fi.replace(snapshot_prefix,'').replace('_iter_','').replace('.caffemodel',''))
                    if ith_iter!=max_ins_at and ith_iter!=max_cls_at:
                        if os.path.exists(fi):
                            os.remove(fi)
                    fi2 = fi.replace('.caffemodel','.solverstate')
                    if os.path.exists(fi2):
                        os.remove(fi2)

        return all_dict


def make_cmd(args):
    if args.jobname=='':
        args.jobname = args.core + '_' + os.path.basename(args.sdict['snapshot_prefix'])
    if args.srun:
        srun = 'srun{} -X -D $PWD --gres gpu:1 '.format(' -p '+args.cluster if args.cluster else '')
        if args.jobname!='none':
            srun += '--job-name='+args.jobname+' '
    else:
        srun=''
    cmd = srun

    cmd += 'python run_test.py --core {} -s {}'.format(args.core, args.solver)
    cmd += '{}'.format(' --clean' if args.clean else '')
    if args.weight:
        cmd += ' --weight {}'.format(args.weight)

    if args.core=='eval_iou':
        if args.logname=='':
            args.logname = args.jobname
        logpath = os.path.join(os.path.dirname(args.solver), args.logname+'.txt')
        if os.path.exists(logpath):
            raise RuntimeError('log already existed:'+logpath)
        if args.logname!='none':
            log = ' 2>/dev/null | tee {}'.format(logpath)
        else:
            log = ''
        cmd += log
    else:
        cmd += ' 2>/dev/null'
    return cmd


def run(args):
    cmd = make_cmd(args)
    args.cmd=cmd
    logger.info('to run:')
    fastprint(cmd)
    if args.dryrun:
        logger.info('dryrun: sleep 20')
        cmd = 'echo PYTHONPATH=$PYTHONPATH; for i in {1..20}; do echo $i; sleep 1; done;'

    my_env = os.environ.copy()
    from base import ADDITIONAL_PYTHONPATH_LIST
    if my_env.has_key('PYTHONPATH'):
        my_env['PYTHONPATH'] = ':'.join(args.additional_pythonpath+ADDITIONAL_PYTHONPATH_LIST)+':'+my_env['PYTHONPATH']
    else:
        my_env['PYTHONPATH'] = ':'.join(args.additional_pythonpath+ADDITIONAL_PYTHONPATH_LIST)
    import subprocess
    THE_JOB = subprocess.Popen(cmd, shell=True, cwd=args.cwd, env=my_env)

    while True:
        retcode=THE_JOB.poll()
        if retcode is not None:
            logger.info('job({}) finished!'.format(args.jobname))
            break

        try:
            time.sleep(1)
        except KeyboardInterrupt:
            THE_JOB.kill()
            logger.info('job({}) killed by CTRL-C!'.format(args.jobname))
            break


def main(args):
    if args.wrapper:
        args.sdict = solver2dict(args.solver)
        run(args)
    else:
        if args.core=='eval_iou':
            ShapeNetTester().evaluate(solver_path=args.solver, clean=args.clean)
        elif args.core=='vis':
            ShapeNetTester().visualize(solver_path=args.solver, args=args)
        else:
            raise ValueError('unknown args.core="{}"'.format(args.core))


def get_args(argv):
    parser = argparse.ArgumentParser(argv[0])

    parser.add_argument('-s','--solver',type=str,
                        help='path to solver')

    parser.add_argument('--clean',dest='clean',action='store_true',default=False,
                        help='clean snapshot folder and only keep best weights')

    parser.add_argument('--cluster',type=str,default='',
                        help='which cluster')
    parser.add_argument('--jobname',type=str,default='',
                        help='cluster job name')
    parser.add_argument('--logname',type=str,default='',
                        help='log file name')

    parser.add_argument('--no-srun',dest='srun',action='store_false',default=True,
                        help='DO NOT use srun')

    parser.add_argument('--dryrun',dest='dryrun',action='store_true',default=False,
                        help='sleep 20 seconds')

    parser.add_argument('--wrapper',dest='wrapper',action='store_true',default=False,
                        help='run with wrapper')

    parser.add_argument('--core',type=str,default='',
                        help='run which core command (vis/eval_iou)')

    parser.add_argument('--weight',type=str,default='',
                        help='using which weight to visualize')

    args = parser.parse_args(argv[1:])
    args.cwd = os.getcwd()
    args.raw_argv = ' '.join(argv)
    args.additional_pythonpath = ['./']
    return args


if __name__ == '__main__':
    args = get_args(sys.argv)
    main(args)
