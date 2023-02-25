# Copyright (C) 2018, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
from pointnet_vanilla import PointNetVanilla
from common import ModelNet10

class PointNetVanilla_mn10(ModelNet10, PointNetVanilla):
    pass

if __name__ == '__main__':
    NETWORK_NAME = os.path.splitext(__file__)[0]
    PointNetVanilla_mn10(NETWORK_NAME).main()
