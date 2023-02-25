# Copyright (C) 2018, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os

from KConly import KConly, KernelCorrelation

class KC32(KConly):
    def sigma(self): return 5e-3
    def num_KC_output(self): return 32
    def num_KC_points(self): return 16
    def KC(self, i='X', o='Xc'):
        return KernelCorrelation(
            self.cc, i, o,'G_indptr','G_indices',
            num_output=self.num_KC_output(),
            num_points_per_kernel=self.num_KC_points(),
            sigma=self.sigma()
        )

if __name__ == '__main__':
    NETWORK_NAME = os.path.splitext(__file__)[0]
    KC32(NETWORK_NAME).main()
