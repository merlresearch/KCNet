# Copyright (C) 2018, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os

from GMonly import GMonly, LocalMaxPooling, GlobalMaxPooling

class GM2t4(GMonly):
    def network(self):
        '''X -> F'''
        cc = self.cc

        cc.comment_bar('Feature')
        XKrelu = lambda i,o,n: cc.XK(i,o,n,axis=-1).relu(o).space()
        XKrelu('X' ,'X1',  64)
        XKrelu('X1','X2',  64)
        XKrelu('X2','X3',  64)
        XKrelu('X3','X4', 128)
        LMPitoj = lambda i,j: LocalMaxPooling(
            cc,i,i+'m','G_indptr','G_indices','G_data'
        ).concat([i+'m',j],i+'m+'+j).space()
        LMPitoj('X2','X4')
        cc.XK('X2m+X4','P',1024,axis=-1).space()
        GlobalMaxPooling(cc, 'P', 'n_offset', 'F')

        self.classifier()

if __name__ == '__main__':
    NETWORK_NAME = os.path.splitext(__file__)[0]
    GM2t4(NETWORK_NAME).main()
