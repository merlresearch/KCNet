# Copyright (C) 2018, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
from common import ModelNet40GraphFull, BaseClassifier, Adam
from kcnet import GlobalMaxPooling, LocalMaxPooling

class GMonly(ModelNet40GraphFull, BaseClassifier, Adam):
    def network(self):
        '''X -> F'''
        cc = self.cc

        cc.comment_bar('Feature')
        XKrelu = lambda i,o,n: cc.XK(i,o,n,axis=-1).relu(o).space()
        XKrelu('X' ,'X1',  64)
        XKrelu('X1','X2',  64)
        XKrelu('X2','X3',  64)
        LocalMaxPooling(cc,'X3','X3m','G_indptr','G_indices','G_data')
        XKrelu('X3m','X4', 128)
        LocalMaxPooling(cc,'X4','X4m','G_indptr','G_indices','G_data')
        cc.XK( 'X4m','P',1024,axis=-1).space()
        GlobalMaxPooling(cc, 'P', 'n_offset', 'F')

        self.classifier()


if __name__ == '__main__':
    NETWORK_NAME = os.path.splitext(__file__)[0]
    GMonly(NETWORK_NAME).main()
