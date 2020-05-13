"""
Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here
import numpy as np
import matplotlib.pyplot as plt
import skimage
from scipy import ndimage
import helper
import submission as sub
import math
import scipy 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
        A = []
        pts1, pts2  = pts1 / M, pts2 / M
        for i in range(pts1.shape[0]):
                x11 = pts1[i,0]
                x12 = pts2[i,0]
                y11 = pts1[i,1]
                y12 = pts2[i,1]
                A.append([x11 *x12, x12*y11, x12, y12*x11, y12*y11, y12, x11, y11, 1])
        A = np.asarray(A)
        u, s, vh = np.linalg.svd(A)
        F = vh[-1,:]
        F = np.reshape(F, [3,3])
        #F = helper._singularize(F)
        F = helper.refineF(F, pts1, pts2)
        T = np.array([[1/M, 0, 0], [0, 1/M, 0], [0, 0, 1]])
        F = np.matmul(np.matmul(np.transpose(T),F), T)
        return F

'''
Q2.2: Seven Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated fundamental matrix.
'''

def sevenpoint(pts1, pts2, M):
        A = []
        pts1, pts2  = pts1 / M, pts2 / M
        for i in range(pts1.shape[0]):
                x11 = pts1[i,0]
                x12 = pts2[i,0]
                y11 = pts1[i,1]
                y12 = pts2[i,1]
                A.append([x11 *x12, x12*y11, x12, y12*x11, y12*y11, y12, x11, y11, 1])
        A = np.asarray(A)      
        u, s, vh = np.linalg.svd(A)
        F1 = vh[-1,:]
        F2 = vh[-2,:]
        F1 = np.reshape(F1, [3,3])
        F2 = np.reshape(F2, [3,3])
        T = np.array([[1/M, 0, 0], [0, 1/M, 0], [0, 0, 1]])
        fun = lambda a: np.linalg.det(a * F1 + (1 - a) * F2)
        a0 = fun(0)
        a1 = (2/3) *(fun(1)-fun(-1)) - (fun(2) - fun(-2))/12
        a2 = 0.5 * fun(1) + 0.5 *fun(-1) - fun(0)
        #a3 = (-1/6) * (fun(1) - fun(-1)) + (fun(2) - fun(-2))/12
        a3 = (fun(1) - fun(1))/2 - a1
        coefficients = [a3, a2, a1, a0] 
        roots = np.roots(coefficients)
        num_roots = len(roots)
        for i in range(len(roots)):
                if np.iscomplex(roots[i]):
                        num_roots = num_roots -1
        
        F = np.zeros([num_roots,3,3])
        T = np.array([[1/M, 0, 0], [0, 1/M, 0], [0, 0, 1]])
        # temp = roots[i] * F1 + (1 - roots[i]) * F2
        # B.append(temp)
        # F[i] = helper.refineF(F[i], pts1, pts2)
        # F[i] = np.matmul(np.matmul(T.T, F[i]), T)

        for i in range(num_roots):
                F[i] = roots[i] * F1 + (1 - roots[i]) * F2
                #F[i] = helper.refineF(F[i], pts1, pts2)
                F[i] = np.matmul(np.matmul(T.T, F[i]), T)
        # print(F[0])
        return F


'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
        #E = np.matmul((K2.T), np.matmul(F, K1)) 
        E = np.matmul(np.matmul(K2.T, F), K1)
        return E
    # Replace pass by your implementation


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
        pt1 = []
        pt2 = []
        for i in range(len(pts1)):
                pt1.append([pts1[i, 0], pts1[i,1], 1])
                pt2.append([pts2[i, 0], pts2[i,1], 1])
        pt1 = np.asarray(pt1)
        pt2 = np.asarray(pt2)
        C11 = C1[0,:]
        C12 = C1[1,:]
        C13 = C1[2,:]
        C21 = C2[0,:]
        C22 = C2[1,:]
        C23 = C2[2,:]
        u1, v1, u2, v2 = pts1[:,0], pts1[:,1], pts2[:,0], pts2[:,1]
        P = np.zeros([len(pts1), 4])
        error = np.zeros([len(pts1), 4])
        sum = 0
        for i in range(len(pts2)):
                D1 = u1[i] * C13 - C11
                D2 = v1[i] * C13 - C12
                D3 = u2[i] * C23 - C21
                D4 = v2[i] * C23 - C22
                A = np.array([D1, D2, D3, D4])
                u, s, vh = np.linalg.svd(A)
                X = vh[-1, :]
                X = X/X[3]
                P[i, :] = X
                pt1_pro = np.matmul(C1, X.T)
                pt1_pro = pt1_pro/pt1_pro[2]
                pt2_pro = np.matmul(C2, X.T)
                pt2_pro = pt2_pro/pt2_pro[2]
                # error1 = pts1[:,i].T - pt1_pro
                # error2 = pt2[:,i].T - pt2_pro
                error1 = pt1[i, :] - pt1_pro
                error2 = pt2[i, :] - pt2_pro
                error = np.linalg.norm(error1) + np.linalg.norm(error2)
                sum += error

                

        # Back Propogation

        # pt1_pro = np.matmul(C1, P.T)
        # pt1_pro = pt1_pro/pt1_pro[2]

        # pt2_pro = np.matmul(C2, P.T)
        # pt2_pro = pt2_pro/pt2_pro[2]
        P = P[:, 0:3] 
        err = sum
        # error1 = pt1.T - pt1_pro
        # error2 = pt2.T - pt2_pro
        # # error = np.square(pt1.T - pt1_pro)+np.square(pt2.T - pt2_pro)
        # # error = error.T
        # err = np.linalg.norm(error1) + np.linalg.norm(error2)
        # #print(P)
        return P,err
 





        

#     # Replace pass by your implementation
#         pass



'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def helper_funct(kerlen = 22, nsig = 3):
        mat = np.zeros((kerlen, kerlen))
        mat[kerlen//2, kerlen//2] = 1
        return ndimage.filters.gaussian_filter(mat, nsig)

def epipolarCorrespondence(im1, im2, F, x1, y1):

    temp = np.zeros([3,1])
    temp[0] = x1
    temp[1] = y1
    temp[2] = 1
    ep_line = np.matmul(F, temp)
    a = ep_line[0]
    b = ep_line[1]
    c = ep_line[2]
   
    max_error = 1e10
    filter_size = 11
    sigma = 3


    kernel = helper_funct(2*filter_size,sigma)
    kernel = np.dstack((kernel, kernel, kernel))

    if (y1 < 474 and y1 > 6):
        mask1 = im1[int(y1-filter_size):int(y1+filter_size), int(x1-filter_size):int(x1+filter_size)]


    error = []
    for y2 in range(y1-30,y1+30):  
        x2 = int((-c - b*y2)/a)

        if (x2-filter_size > 0 and (x2+filter_size < im2.shape[1])) and (y2-filter_size>0 and (y2+filter_size<=im2.shape[0])):
            mask2 = im2[int(y2-filter_size):int(y2+filter_size), int(x2-filter_size):int(x2+filter_size)]
            distance = mask1 - mask2
            diff = np.multiply(kernel, distance)
            err = np.linalg.norm(diff)
            error.append(err)
        
            if err < max_error:
                max_error = err

        return x2, y2

'''
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
'''
def ransacF(pts1, pts2, M):
    final_F= 0
    max_inliers= 0
    num_iter= 1500
    tol= 0.0009
    num_inliers= 0
    N = pts1.shape[0]
    ones = np.ones([N,1])
    pts1_in= pts1
    pts2_in= pts2
    pts1_org=  np.concatenate((pts1, ones), axis=1)
    pts2_org=  np.concatenate((pts2, ones), axis=1)

    for k in range(0,num_iter):
        pts1= []
        pts2= []
        pt_indices_list= [np.random.randint(0, N-1) for p in range(0, 7)]
        for i in range(0,7):
            pts1.append(pts1_in[pt_indices_list[i],:])
            pts2.append(pts2_in[pt_indices_list[i],:])
        pts1_mat= np.vstack(pts1)
        pts2_mat= np.vstack(pts2)
        F7= sevenpoint(pts1_mat,pts2_mat, M)
        for j in range(0,len(F7)):
            num_inliers= 0
            for k in range(0,pts1_org.shape[0]):
                error_x= np.abs(np.matmul(np.matmul(np.transpose(pts2_org[k]), F7[j]), pts1_org[k])) 
                if (error_x < tol):
                    num_inliers= num_inliers +1
            if(num_inliers > max_inliers):
                max_inliers= num_inliers
                final_F= F7[j]
    return final_F

def ransacF_bundle(pts1, pts2, M):
    bestF= 0
    max_inliers= 0
    num_iter= 2000
    tol= 0.001
    num_inliers= 0
    N = pts1.shape[0]
    ones = np.ones([N,1])
    pts1_in= pts1
    pts2_in= pts2
    pts1_org=  np.concatenate((pts1, ones), axis=1)
    pts2_org=  np.concatenate((pts2, ones), axis=1)

    for k in range(0,num_iter):
        pts1= []
        pts2= []
        pt_indices_list= [np.random.randint(0, N-1) for p in range(0, 7)]
        for i in range(0,7):
            pts1.append(pts1_in[pt_indices_list[i],:])
            pts2.append(pts2_in[pt_indices_list[i],:])
        pts1_mat= np.vstack(pts1)
        pts2_mat= np.vstack(pts2)
        F7= sevenpoint(pts1_mat,pts2_mat, M)
        for j in range(0,len(F7)):
            num_inliers= 0
            a = []
            b = []
            for k in range(0,pts1_org.shape[0]):
                error_x= np.abs(np.matmul(np.matmul(np.transpose(pts2_org[k]), F7[j]), pts1_org[k])) 
                if (error_x < tol):
                    num_inliers= num_inliers +1
                    a.append(pts1_in[k])
                    b.append(pts2_in[k])
            if(num_inliers > max_inliers):
                max_inliers= num_inliers
                bestF= F7[j]
                c = a
                d = b
    return bestF,c,d

"""
for i in range(len(F7)):
    E = sub.essentialMatrix(F7[i], K1, K2) 
    M2 = helper.camera2(E)
    C1 = np.concatenate([np.eye(3), np.zeros([3,1])], axis = 1)
    C1 = np.matmul(K1, C1)
    for k in range(4):
        F = sub.ransacF(data['pts1'], data['pts2'], M2[:,:,k])
"""

'''
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
        theta = np.linalg.norm(r)
        u = r/theta 
        S = np.array([ [   0 ,  - u[2], u[1]], 
                        [ u[2],    0 ,  -u[0]],
                        [  -u[1], u[0],    0 ]])
        R = np.cos(theta)*np.eye(3) + np.sin(theta)*S + (1 - np.cos(theta))* np.matmul(u, u.T)
        return R

'''
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
        A  =  (R - R.T)/2
        r = np.asarray([A[2,1], A[0,2], A[1,0]]).T
        s = np.linalg.norm(r)
        c = (R[0,0] + R[1,1] + R[2,2]-1)/2
        theta = np.arctan2(s,c)
        

        if s==0 and c==1:
                return np.asarray([0,0,0])

        elif s==0 and c==-1:
                u = r/s
                v= (R + np.eye(3)).reshape(9,1)
                u = v/np.linalg.norm(v,2)
                r = u*np.pi
                if np.linalg.norm(r,2)==np.pi and ( (r[0] ==0 and r[1] ==0 and r[2]<0) or ( r[0]==0 and r[1]<0 ) or (r[0]<0) ):
                        return -r
                else:
                        return r

        elif np.sin(theta) != 0:
                u = r/s
                return  u*theta

        else:
                print('No condition satisfied')
                return None
        return r

'''
Q5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
        t2 = x[len(x) - 3:]
        t2 = np.array([[t2[0]],[t2[1]],[t2[2]]])
        r = x[len(x)-6:len(x) - 3]
        P = x[:len(x)-6]
        R2 =rodrigues(r)                #change the implementation after getting the right one
        M2 = np.concatenate((R2, t2), axis = 1)
        # print(M2)
        # print(M1)
        C1 = np.matmul(K1, M1)
        C2 = np.matmul(K2, M2)
        P_standard = np.reshape(P, (len(P)/3,3))
        # print(P_standard)
        pt1 = []
        for i in range(len(P_standard)):
                pt1.append([P_standard[i, 0], P_standard[i,1], P_standard[i,2], 1])
        P_Homo = np.array(pt1)
        # print(P_Homo)
        p1_hat_homo = []
        p2_hat_homo = []
        # print(P_Homo[0,:])
        for i in range(len(P_Homo)):
                p1_hat_homo.append(np.matmul(C1, P_Homo[i, :]))
                p2_hat_homo.append(np.matmul(C2, P_Homo[i, :]))
                temp_p1 = np.matmul(C1, P_Homo[i, :])
                temp_p1 = temp_p1/temp_p1[2]
                temp_p2 = np.matmul(C2, P_Homo[i, :])
                temp_p2 = temp_p2/temp_p2[2]
        p1_hat_homo = np.array(p1_hat_homo)
        p2_hat_homo = np.array(p2_hat_homo)
        # print(np.shape(p1_hat_homo))
        #print(p2_hat_homo)
        # print('Ho')
        # print(p1_hat_homo[:]/p1_hat_homo[:, 2])
        for k in range(len(p1_hat_homo)):
                p1_hat_homo[k] = p1_hat_homo[k]/p1_hat_homo[k, 2]
                p2_hat_homo[k] = p2_hat_homo[k]/p2_hat_homo[k, 2]
        # print(p1_hat_homo)
        # print(p2_hat_homo)
        p2_hat = p2_hat_homo[:, :2]
        p1_hat = p1_hat_homo[:, :2]
        # print(p1-p1_hat)
        final_error = 0
        for i in range(len(p2_hat)):
                error1 = p1[i, :] - p1_hat[i, :]
                error2 = p2[i, :] - p2_hat[i, :]
                error = np.linalg.norm(error1) + np.linalg.norm(error2)
                final_error += error
        #print(final_error)
        residual = np.concatenate([(p1-p1_hat).reshape([-1]),(p2-p2_hat).reshape([-1])])
        # residual = residual.flatten()
        # print(np.shape(residual))
        #print(residual)
    # Replace pass by your implementation
        return residual

'''
Q5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
        P = P_init
        R2 = M2_init[:, 0:3]
        t2 = M2_init[:, 3]
        #change the below latter
        r2 = invRodrigues(R2)
        r2 = np.reshape(r2, [1,3]) 
        x = np.concatenate([P.reshape([-1]), r2[-1], t2])
        args = x, K1, M1, p1, K2, p2
        func = lambda args : (rodriguesResidual(K1,M1,p1,K2,p2,x)**2).sum()
        x = scipy.optimize.least_squares(func, x)
        x = x.x
        t2 = x[len(x) - 3:]
        t2 = np.array([[t2[0]],[t2[1]],[t2[2]]])
        r = x[len(x)-6:len(x) - 3]
        P = x[:len(x)-6]
        P = np.reshape(P, [len(P)/3, 3])
        R2 = rodrigues(r)                #change the implementation after getting the right one
        M2 = np.concatenate((R2, t2), axis = 1)
        # for i in range(len(p2_hat)):
        #         error1 = p1[i, :] - p1_hat[i, :]
        #         error2 = p2[i, :] - p2_hat[i, :]
        #         error = np.linalg.norm(error1) + np.linalg.norm(error2)
        #         final_error += error
        # print(final_error)
        return M2, P


if __name__ == "__main__":      
        im1 = plt.imread('../data/im1.png')
        im2 = plt.imread('../data/im2.png')
        M = 640        
        data = np.load('../data/some_corresp_noisy.npz')
        r2 = np.ones(3)
        t2 = np.ones(3)
        M = 640
        F7, c, d = sub.ransacF_bundle(data['pts1'], data['pts2'], M)
        pts1_inliers = np.array(c)
        pts2_inliers = np.array(d)
        M1 = np.concatenate([np.eye(3), np.zeros([3,1])], axis = 1)


        intrinsics = np.load('../data/intrinsics.npz')
        K1, K2 = intrinsics.f.K1 , intrinsics.f.K2
        E = sub.essentialMatrix(F7, K1, K2)
        M2 = helper.camera2(E)      #M2 has the shape of [4,4,3] so be careful
        M1 = np.concatenate([np.eye(3), np.zeros([3,1])], axis = 1)
        C1 = np.matmul(K1, M1)
        err_min = 10000
        for i in range(4):
                C2 = np.matmul(K2, M2[:,:,i])
                P, err =  sub.triangulate(C1, data['pts1'], C2, data['pts2'])
                if np.all(P[:,2] > 0):
                        sol = M2[:,:,i]
                        C2 = np.matmul(K2, M2[:,:,i])
                        break
        M2 = sol
        C2 = np.matmul(K2, M2)
        P, err = triangulate(C1, pts1_inliers, C2, pts2_inliers)
        # print(P)
        R2 = M2[:, 0:3]
        t2 = M2[:, 3]
        print('The Initial value of M2')
        #change the below latter
        r2 =invRodrigues(R2)
        r2 = np.reshape(r2, [1,3])
        # print(M2) 
        x = np.concatenate([P.reshape([-1]), r2[-1], t2])
        residuals = sub.rodriguesResidual(K1, M1, pts1_inliers, K2, pts2_inliers, x)
        M2, P2 = sub.bundleAdjustment(K1, M1, pts1_inliers, K2, M2, pts2_inliers, P)
        # print('The Optimized value of M2')
        # print(P2)
        #print(P2)



        fig = plt.figure(1)
        #ax = Axes3D(fig)
        ax = fig.add_subplot(111, projection = '3d')

        x = P2[:,0]
        y = P2[:,1]
        z = P2[:,2]
        plt.gca().set_aspect('equal',adjustable = 'box')
        # ax.scatter(x,y,z, color='blue')

        for i in range(P.shape[0]):
                ax.scatter(P2[i][0], P2[i][1], P2[i][2], c='r', marker='o')
                #ax.scatter(P[i][0], P[i][1], P[i][2], c='b', marker='o')
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        plt.show()
        plt.close()


        



        # r = np.array([0.01,0.25,0.05])
        # R = sub.rodrigues(r)
        # print(R)
        # print(invRodrigues(R))