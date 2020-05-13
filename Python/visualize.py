'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize
import cv2 as cv 
import submission as sub
import helper
import random
import findM2

data = np.load('../data/some_corresp.npz')
selected_points = np.load('../data/templeCoords.npz')
camera_cal = np.load('../data/intrinsics.npz')

im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')

rows = im1.shape[0]
cols = im1.shape[1]
M = max(rows,cols)

K1 = camera_cal['K1']
K2 =  camera_cal['K2']
M1 = np.array([[1, 0, 0,0], [0, 1, 0,0],[0,0,1,0]])

F8 = sub.eightpoint(data['pts1'], data['pts2'], M)
print(np.linalg.matrix_rank(F8))
E = sub.essentialMatrix(F8, K1, K2)

M2s = helper.camera2(E)

C1 = np.matmul(K1, M1)
C2s = []
for j in range(4):
	temp = np.matmul(K2, M2s[:,:,j])
	C2s.append(temp)
C2s = np.asarray(C2s)

N = 288
x1_selected = selected_points['x1']
y1_selected = selected_points['y1']
selected_points_1 = np.zeros([N,2])
for i in range(N):
	selected_points_1[i][0] = x1_selected[i]
	selected_points_1[i][1] = y1_selected[i]
selected_points_1 = np.asarray(selected_points_1)

# Finding corresponding points
N = len(x1_selected)
x2_selected = []
y2_selected = []
for i in range(N):
	x_t, y_t = sub.epipolarCorrespondence(im1, im2, F8, int(x1_selected[i]), int(y1_selected[i]))
	x2_selected.append(x_t)
	y2_selected.append(y_t)
x2_selected = np.asarray(x2_selected)
y2_selected = np.asarray(y2_selected)

selected_points_2 = np.zeros([N,2])
for i in range(N):
	selected_points_2[i][0] = x2_selected[i]
	selected_points_2[i][1] = y2_selected[i]
selected_points_2 = np.asarray(selected_points_2)

#P, err = sub.triangulate(C1, selected_points_1, C2s[0], selected_points_2)
M2, C2, P = findM2.find(M2s, C1, selected_points_1, selected_points_2, K2)
# np.savez('../results/q4_2.npz', M1 = M1, M2 = M2, C1 = C1, C2 = C2)

fig = plt.figure()
# ax = Axes3D(fig)
ax = fig.add_subplot(111, projection = '3d')

x = P[:,0]
y = P[:,1]
z = P[:,2]
plt.gca().set_aspect('equal',adjustable = 'box')
# ax.scatter(x,y,z, color='blue')

for i in range(P.shape[0]):
	ax.scatter(P[i][0], P[i][1], P[i][2], c='r', marker='o')

ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')

np.savez('q4_2', F = F8, M1 = M1, M2 = M2, C1 = C1, C2 = C2)

plt.show()
plt.close()