# Copyright (C) 2018, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
from KC32GM2t4 import KC32GM2t4
from common import ModelNet10GraphSimple

class KC32GM2t4_mn10(ModelNet10GraphSimple, KC32GM2t4):
    pass

if __name__ == '__main__':
    NETWORK_NAME = os.path.splitext(__file__)[0]
    KC32GM2t4_mn10(NETWORK_NAME).main()
