# Copyright (C) 2018, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
from common import ShapeNet18nn, BaseGraphDataFull, BaseClassifierWithDropout, Adam
from kcnet import LocalMaxPooling

class GM2t4(BaseGraphDataFull, ShapeNet18nn, BaseClassifierWithDropout, Adam):
    def LMPitoj(self, i, j):
        return LocalMaxPooling(
            self.cc,i,i+'m','G_indptr','G_indices','G_data'
        ).concat([i+'m',j],i+'m+'+j).space()

    def network(self):
        cc = self.cc

        cc.comment_bar('Feature')
        XKrelu = lambda i,o,n: cc.XK(i,o,n,axis=-1).relu(o).space()
        XKrelu('X' ,'X1',  64)
        XKrelu('X1','X2',  64)
        XKrelu('X2','X3', 128)
        XKrelu('X3','X4', 128)
        self.LMPitoj('X2','X4')
        XKrelu('X2m+X4','X5', 512)
        cc.XK( 'X5','X6',1024,axis=-1).space() #(B*N)xK

        self.global_pool_and_tile()

        cc.concat(['X1','X2','X3','X4','X5','H_tile'],'F',axis=-1)

        self.classifier()


if __name__ == '__main__':
    NETWORK_NAME = os.path.splitext(__file__)[0]
    GM2t4(NETWORK_NAME).main()
