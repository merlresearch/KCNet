# Copyright (C) 2018, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import caffecup
from string import Template

def KernelCorrelation(
        cc,
        input, output,
        G_indptr, G_indices,
        num_output, num_points_per_kernel,
        sigma, kernel_filler=''
):
    assert(isinstance(cc, caffecup.Designer))
    shape = cc.shape_of(input)
    shape[-1] = num_output
    assert(cc.register_new_blob(output, shape))
    cc.shape_of(G_indptr)
    cc.shape_of(G_indices)
    if kernel_filler=='':
        kernel_filler = caffecup.design.filler_uniform(-0.2,0.2)
    kernel_filler = 'kernel_'+kernel_filler

    s = Template(
'''layer {
  name: "KC" type: "KernelCorrelation"
  bottom: "$input"
  bottom: "$G_indptr"
  bottom: "$G_indices"
  top: "$output"
  kernel_correlation_param {
   num_output:$num_output num_points_per_kernel:$num_points_per_kernel
   sigma:$sigma $kernel_filler
  }
}
'''
    )
    cc.fp.write(s.substitute(locals()))
    return cc

def GlobalMaxPooling(
        cc,
        input, input_offset, output,
        name='GlobalMaxPool'
):
    assert(isinstance(cc, caffecup.Designer))
    shape = cc.shape_of(input)
    shape_offset = cc.shape_of(input_offset)
    shape[0] = shape_offset[0]
    assert(cc.register_new_blob(output, shape))

    s = Template(
'''layer {
  name: "$name" type: "GraphPooling"
  graph_pooling_param { mode: MAX }
  bottom: "$input"
  bottom: "$input_offset"
  propagate_down: true
  propagate_down: false
  top: "$output"
}
'''
    )
    cc.fp.write(s.substitute(locals()))
    return cc

def LocalGraphPooling(
        cc,
        input, output,
        G_indptr, G_indices, G_data,
        mode, name
):
    assert(mode in ['MAX','AVE'])
    assert(isinstance(cc, caffecup.Designer))
    cc.shape_of(G_indptr)
    cc.shape_of(G_indices)
    assert(cc.register_new_blob(output, cc.shape_of(input)))

    s = Template(
'''layer {
  name: "$name" type: "GraphPooling"
  graph_pooling_param { mode: $mode }
  bottom: "$input"
  bottom: "$G_indptr" bottom: "$G_indices" bottom: "$G_data"
  propagate_down: true
  propagate_down: false propagate_down: false propagate_down: false
  top: "$output"
}
'''
    )
    cc.fp.write(s.substitute(locals()))
    return cc

def LocalMaxPooling(
        cc,
        input, output,
        G_indptr, G_indices, G_data,
        name=''
):
    assert(isinstance(cc, caffecup.Designer))
    if not hasattr(cc, 'n_LMP'):
        cc.n_LMP=1
    else:
        cc.n_LMP+=1
    if name=='':
        name = 'LocalMaxPool{:d}'.format(cc.n_LMP)

    return LocalGraphPooling(cc, input, output,
                             G_indptr, G_indices, G_data,
                             mode='MAX', name=name)

def LocalAvePooling(
        cc,
        input, output,
        G_indptr, G_indices, G_data,
        name=''
):
    assert(isinstance(cc, caffecup.Designer))
    if not hasattr(cc, 'n_LAP'):
        cc.n_LAP=1
    else:
        cc.n_LAP+=1
    if name=='':
        name = 'LocalAvePool{:d}'.format(cc.n_LAP)

    return LocalGraphPooling(cc, input, output,
                             G_indptr, G_indices, G_data,
                             mode='AVE', name=name)
