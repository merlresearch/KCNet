# Copyright (C) 2018, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
from GM2t4 import GM2t4
from KC32 import KC32

class KC32GM2t4(GM2t4, KC32):
    def network(self):
        cc = self.cc

        cc.comment_bar('Feature')
        self.KC()
        cc.concat(['X','Xc'],'Xcat').space()

        XKrelu = lambda i,o,n: cc.XK(i,o,n,axis=-1).relu(o).space()
        XKrelu('Xcat' ,'X1',  64)
        XKrelu('X1','X2',  64)
        XKrelu('X2','X3', 128)
        XKrelu('X3','X4', 128)
        self.LMPitoj('X2','X4')
        XKrelu('X2m+X4','X5', 512)
        cc.XK( 'X5','X6',1024,axis=-1).space() #(B*N)xK

        self.global_pool_and_tile()

        cc.concat(['Xcat','X1','X2','X3','X4','X5','H_tile'],'F',axis=-1)

        self.classifier()


if __name__ == '__main__':
    NETWORK_NAME = os.path.splitext(__file__)[0]
    KC32GM2t4(NETWORK_NAME).main()
