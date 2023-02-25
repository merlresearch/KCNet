# Copyright (C) 2018, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
from common import ShapeNet18nn, BaseGraphDataSimple, BaseClassifierWithDropout, Adam
from kcnet import KernelCorrelation

class KConly(BaseGraphDataSimple, ShapeNet18nn, BaseClassifierWithDropout, Adam):
    def KC(self, i='X', o='Xc'):
        KernelCorrelation(self.cc, i, o,'G_indptr','G_indices',
                          num_output=16,num_points_per_kernel=self.knn(),sigma=5e-3)
        return self.cc

    def network(self):
        cc = self.cc

        cc.comment_bar('Feature')
        self.KC()
        cc.concat(['X','Xc'],'Xcat').space()

        XKrelu = lambda i,o,n: cc.XK(i,o,n,axis=-1).relu(o).space()
        XKrelu('Xcat','X1',64)
        XKrelu('X1','X2',  64)
        XKrelu('X2','X3', 128)
        XKrelu('X3','X4', 128)
        XKrelu('X4','X5', 512)
        cc.XK( 'X5','X6',1024,axis=-1).space() #(B*N)xK

        self.global_pool_and_tile()

        cc.concat(['Xcat','X1','X2','X3','X4','X5','H_tile'],'F',axis=-1)

        self.classifier()


if __name__ == '__main__':
    NETWORK_NAME = os.path.splitext(__file__)[0]
    KConly(NETWORK_NAME).main()
