# Copyright (C) 2018, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
from KC32 import KC32
from kcnet import GlobalMaxPooling

class KC32t3(KC32):
    def network(self):
        cc = self.cc

        cc.comment_bar('Feature')
        XKrelu = lambda i,o,n: cc.XK(i,o,n,axis=-1).relu(o).space()
        XKrelu('X' ,'X1',  64)
        XKrelu('X1','X2',  64)
        XKrelu('X2','X3',  64)
        self.KC('X','Xc').concat(['X3','Xc'],'Xcat').space()
        XKrelu('Xcat','X4', 128)
        cc.XK( 'X4','P',1024,axis=-1).space()
        GlobalMaxPooling(cc, 'P', 'n_offset', 'F')

        self.classifier()

if __name__ == '__main__':
    NETWORK_NAME = os.path.splitext(__file__)[0]
    KC32t3(NETWORK_NAME).main()
