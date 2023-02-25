# Copyright (C) 2018, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
from common import ModelNet40, BaseClassifier, Adam

class PointNetVanilla(ModelNet40, BaseClassifier, Adam):
    def network(self):
        '''X -> F'''
        cc = self.cc
        N = self.train_data_shape()[1]

        cc.comment_bar('Feature')
        XKrelu = lambda i,o,n: cc.XK(i,o,n,axis=-1).relu(o).space()
        XKrelu('X' ,'X1',  64)
        XKrelu('X1','X2',  64)
        XKrelu('X2','X3',  64)
        XKrelu('X3','X4', 128)
        cc.XK( 'X4','P',1024,axis=-1).space()
        cc.reshape('P','P_Bx1xNxK',shape=[0,1,N,-1])
        cc.pool('P_Bx1xNxK','F',kernel_h=N,kernel_w=1,name='global_max_pool')

        self.classifier()


if __name__ == '__main__':
    NETWORK_NAME = os.path.splitext(__file__)[0]
    PointNetVanilla(NETWORK_NAME).main()
