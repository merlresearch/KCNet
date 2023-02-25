# Copyright (C) 2018, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import sys
import glob
import time
import argparse

import numpy as np
from matplotlib import pyplot as plt

def uniform_sphere_points(n):
    golden_angle = np.pi * (3 - np.sqrt(5))
    theta = golden_angle * np.arange(n)
    z = np.linspace(1 - 1.0 / n, 1.0 / n - 1, n)
    radius = np.sqrt(1 - z * z)

    points = np.zeros((n, 3))
    points[:,0] = radius * np.cos(theta)
    points[:,1] = radius * np.sin(theta)
    points[:,2] = z
    return points


def uniform_rand_rotation(n):
    zdirs = uniform_sphere_points(n)
    Rs=[]
    for i in xrange(n):
        zdir=zdirs[i,:]
        zdir = zdir / np.linalg.norm(zdir)
        rdir = np.random.rand(3,)*2.0-1.0
        rdir = rdir / np.linalg.norm(rdir)
        ydir = np.cross(zdir, rdir)
        ydir = ydir / np.linalg.norm(ydir)
        xdir = np.cross(ydir, zdir)
        xdir = xdir / np.linalg.norm(xdir)
        Rs.append(
            [xdir,ydir,zdir]
        )
    Rs=np.array(Rs)
    return Rs


def skew(v):
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def rand_ortho_rotation_matrix():
    k = np.zeros((3,), dtype=int)
    k[np.random.randint(0,3)]=1 if np.random.rand()>0.5 else -1
    K = skew(k)

    all_theta = [0, 90, 180, 270]
    theta = np.deg2rad(all_theta[np.random.randint(0,4)])

    R = np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*np.dot(K,K)
    return R.astype(int)


def rand_rotation_matrix(deflection=1.0, randnums=None):
    """
    Creates a random rotation matrix.

    deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
    rotation. Small deflection => small perturbation.
    randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.
    """
    # from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c

    if randnums is None:
        randnums = np.random.uniform(size=(3,))

    theta, phi, z = randnums

    theta = theta * 2.0*deflection*np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0*np.pi  # For direction of pole deflection.
    z = z * 2.0*deflection  # For magnitude of pole deflection.

    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.

    r = np.sqrt(z)
    Vx, Vy, Vz = V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
        )

    st = np.sin(theta)
    ct = np.cos(theta)

    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

    # Construct the rotation matrix  ( V Transpose(V) - I ) R.

    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M


def main(args):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(sys.argv[0])

    args = parser.parse_args(sys.argv[1:])
    args.script_folder = os.path.dirname(os.path.abspath(__file__))

    main(args)
