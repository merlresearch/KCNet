# Copyright (C) 2018, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
from common import ShapeNet18nn, BaseClassifierWithDropout, Adam

class PointNet(ShapeNet18nn, BaseClassifierWithDropout, Adam):
    def network(self):
        '''X -> F'''
        cc = self.cc
        N = self.train_data_shape()[1]

        cc.comment_bar('Feature')
        XKrelu = lambda i,o,n: cc.XK(i,o,n,axis=-1).relu(o).space()
        XKrelu('X' ,'X1',  64)
        XKrelu('X1','X2',  64)
        XKrelu('X2','X3', 128)
        XKrelu('X3','X4', 128)
        XKrelu('X4','X5', 512)
        cc.XK( 'X5','X6',1024,axis=-1).space() #BxNxK

        cc.reshape('X6','X6_Bx1xNxK',shape=[0,1,N,-1])
        cc.pool('X6_Bx1xNxK','P_pre',kernel_h=N,kernel_w=1,name='global_max_pool') #Bx1x1xK
        cc.reshape('P_pre','P',shape=[0,-1]) #BxK
        cc.concat(['P','onehot'],'H_pre',axis=-1) #Bx(K+16)
        cc.reshape('H_pre','H',shape=[0,1,-1]) #Bx1x(K+16)
        cc.tile('H','H_tile',axis=1,tiles=N).space() #BxNx(K+16)

        cc.concat(['X1','X2','X3','X4','X5','H_tile'],'F',axis=-1)

        self.classifier()


if __name__ == '__main__':
    NETWORK_NAME = os.path.splitext(__file__)[0]
    PointNet(NETWORK_NAME).main()
