# Copyright (C) 2018, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
from KC32GM2t4 import KC32GM2t4

class KC32GM2t4_M4sigma1en2(KC32GM2t4):
    def sigma(self): return 1e-2
    def num_KC_points(self): return 4

if __name__ == '__main__':
    NETWORK_NAME = os.path.splitext(__file__)[0]
    KC32GM2t4_M4sigma1en2(NETWORK_NAME).main()
