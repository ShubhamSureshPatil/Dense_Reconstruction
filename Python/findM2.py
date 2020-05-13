'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, p1, p2, R and P to q3_3.mat
'''

import numpy as np
import matplotlib.pyplot as plt
import submission as sub
import helper


data = np.load('../data/some_corresp.npz')

M = 640

F8 = sub.eightpoint(data['pts1'], data['pts2'], M)
assert F8.shape == (3, 3), 'eightpoint returns 3x3 matrix'

intrinsics = np.load('../data/intrinsics.npz')
K1, K2 = intrinsics.f.K1 , intrinsics.f.K2
E = sub.essentialMatrix(F8, K1, K2)
M2 = helper.camera2(E)      #M2 has the shape of [4,4,3] so be careful
C1 = np.concatenate([np.eye(3), np.zeros([3,1])], axis = 1)
C1 = np.matmul(K1, C1)
err_min = 10000
for i in range(4):
    C2 = np.matmul(K2, M2[:,:,i])
    P, err =  sub.triangulate(C1, data['pts1'], C2, data['pts2'])
    if np.all(P[:,2] > 0):
        sol = M2[:,:,i]
        C2 = np.matmul(K2, M2[:,:,i])
        break

M2 = sol
np.savez('q3_3.npz', M2 = M2, C2 = C2, P = P)

# C1 = np.matmul(K1, C1)
# C2 = np.matmul(K2, M2)


def find(M2, C1, selected_points_1, selected_points_2, K2):
    for i in range(4):
        C2 = np.matmul(K2, M2[:,:,i])
        P, err =  sub.triangulate(C1, selected_points_1, C2, selected_points_2)
        if np.all(P[:,2] > 0):
            sol = M2[:,:,i]
            C2 = np.matmul(K2, M2[:,:,i])
            break
    return sol, C2, P