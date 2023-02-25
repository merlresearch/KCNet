# Copyright (C) 2018, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
from KConly import KConly
from kcnet import KernelCorrelation

class KC32(KConly):
    def KC(self, i='X', o='Xc'):
        KernelCorrelation(self.cc, i, o,'G_indptr','G_indices',
                          num_output=32,num_points_per_kernel=self.knn(),sigma=5e-3)
        return self.cc


if __name__ == '__main__':
    NETWORK_NAME = os.path.splitext(__file__)[0]
    KC32(NETWORK_NAME).main()
