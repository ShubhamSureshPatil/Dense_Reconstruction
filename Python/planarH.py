import numpy as np
import cv2
from BRIEF import briefLite, briefMatch


def computeH(p1, p2):   #initially it was def computeH(p1,p2)
    '''
    INPUTS:
    p1 and p2 - Each are size (2 x N) matrices of corresponding (x, y)'  
                 coordinates between two images
    OUTPUTS:
    H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear 
            equation
    '''
    assert(p1.shape[1]==p2.shape[1])
    assert(p1.shape[0]==2)
    p2, p1 = p1, p2
    
    a = []
    for i in range(0,4):
        u = p1[0][i]
        v = p1[1][i]
        x = p2[0][i]
        y = p2[1][i]

        l1 = [0,0,0, -u, -v, -1, u * y, v*y, y]
        l2 = [u, v, 1, 0, 0, 0, -u * x, -x * v, -x]
        a.append(l1)
        a.append(l2)
    b = np.matrix(a)
    
    #svd decomposition
    u, s, vh = np.linalg.svd(b)
    vh = np.transpose(vh)
    H2to1 = np.reshape(vh[:,8], (3, 3))
    H2to1 = H2to1/(H2to1.item(8))
    return H2to1

def ransacH(matches, locs1, locs2, num_iter=5000, tol=2):
    '''
    Returns the best homography by computing the best set of matches using
    RANSAC
    INPUTS
        locs1 and locs2 - matrices specifying point locations in each of the images
        matches - matrix specifying matches between these two sets of point locations
        nIter - number of iterations to run RANSAC
        tol - tolerance value for considering a point to be an inlier
        
    OUTPUTS
        bestH - homography matrix with the most inliers found during RANSAC
    ''' 
    ###########################
    p1 = locs1[matches[:,0], 0:2]
    p2 = locs2[matches[:,1], 0:2]
    maximum = 0

    for i in range(num_iter):
        var1 = np.random.randint(0, len(matches))
        var2 = np.random.randint(0, len(matches))
        var3 = np.random.randint(0, len(matches))
        var4 = np.random.randint(0, len(matches))
        x1 =  p1[var1]
        x2 =  p1[var2]
        x = np.vstack((x1, x2))
        x3 =  p1[var3]
        x = np.vstack((x, x3))
        x4 =  p1[var4]
        x = np.vstack((x,x4))
        y1 =  p2[var1]
        y2 =  p2[var2]
        y = np.vstack((y1,y2))
        y3 =  p2[var3]
        y = np.vstack((y, y3))
        y4 =  p2[var4]
        y = np.vstack((y, y4))
        h = computeH(x.T, y.T)
        count = 0
        for k in range(len(matches)):
            q = p1[k]
            r = p2[k]
            
            q = np.transpose([q[0], q[1], 1])
            r = np.transpose([r[0], r[1], 1])
            h2 = np.linalg.inv(h)
            h2 = h2/(h2.item(8))
            r2 = np.matmul(h2, q.T)
            r2 = np.float64(r2)
            if(r2.item(2) == 0):
                print('Division by 0 encountered')
                continue   
            r2 = r2/(r2.item(2))
            error = r - r2 
            error = np.linalg.norm(r2-r)
            if (error < tol):
                count += 1
        if (count > maximum):
            maximum = count
            finalH = h            
    print('maxInliers = ', maximum)
    print(finalH)
    maxInliers = maximum
    return finalH, maxInliers


if __name__ == '__main__':
    im1 = cv2.imread('../data/model_chickenbroth.jpg')
    im2 = cv2.imread('../data/chickenbroth_01.jpg')
#    locs1, desc1 = briefLite(im1)
#    locs2, desc2 = briefLite(im2)
    locs1 = np.load('locs1_planar.npy')
    locs2 = np.load('locs2_planar.npy')
#    matches = briefMatch(desc1, desc2)
    matches = np.load('matches_planar.npy')
    p1 = locs1[matches[:, 0], 0:2]
    p2 = locs2[matches[:, 1], 0:2]
    ransacH(matches, locs1, locs2, num_iter=5000, tol= 2)

