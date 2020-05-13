from skimage import feature, color, transform, io
import matplotlib.pyplot as plt
import numpy as np
import logging
import sys, os
import scipy
import cv2

def vis_edgelets(image, edgelets, col, show=True):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    locations, directions, strengths = edgelets
    for i in range(locations.shape[0]):
        xax = [locations[i, 0] - directions[i, 0] * strengths[i] / 2,
               locations[i, 0] + directions[i, 0] * strengths[i] / 2]
        yax = [locations[i, 1] - directions[i, 1] * strengths[i] / 2,
               locations[i, 1] + directions[i, 1] * strengths[i] / 2]
        # print (xax, yax)
        # sys.exit()
        plt.plot(xax, yax, col)

    if show:
        plt.show()


def compute_edgelets(image, sigma=3):
    gray_img = color.rgb2gray(image)
    edges = feature.canny(gray_img, sigma)
    # print (edges.shape, "edges from feature")
    lines = transform.probabilistic_hough_line(edges, line_length=3,
                                               line_gap=2)

    # print (len(lines), "lines from hough")
    # print (lines[0])
    locations = []
    directions = []
    strengths = []

    for p0, p1 in lines:
        p0, p1 = np.array(p0), np.array(p1)
        locations.append((p0 + p1) / 2) # computes the average of the starting point and ending point
        directions.append(p1 - p0)
        strengths.append(np.linalg.norm(p1 - p0))

    # convert to numpy arrays and normalize
    locations = np.array(locations)
    directions = np.array(directions)
    strengths = np.array(strengths)

    directions = np.array(directions) / \
        np.linalg.norm(directions, axis=1)[:, np.newaxis]

    return (locations, directions, strengths)


def remove_inliers(model, edgelets, threshold_inlier=10):

    inliers = compute_votes(edgelets, model, 10) > 0
    locations, directions, strengths = edgelets
    locations = locations[~inliers]
    directions = directions[~inliers]
    strengths = strengths[~inliers]
    edgelets = (locations, directions, strengths)
    return edgelets

def ransac_vanishing_point(edgelets, num_ransac_iter=2000, threshold_inlier=5):

    locations, directions, strengths = edgelets
    lines = edgelet_lines(edgelets)

    num_pts = strengths.size

    arg_sort = np.argsort(-strengths)
    first_index_space = arg_sort[:num_pts // 5]
    second_index_space = arg_sort[:num_pts // 2]

    best_model = None
    best_votes = np.zeros(num_pts)

    for ransac_iter in range(num_ransac_iter):
        ind1 = np.random.choice(first_index_space)
        ind2 = np.random.choice(second_index_space)

        l1 = lines[ind1]
        l2 = lines[ind2]

        current_model = np.cross(l1, l2)

        if np.sum(current_model**2) < 1 or current_model[2] == 0:
            # reject degenerate candidates
            continue

        current_votes = compute_votes(
            edgelets, current_model, threshold_inlier)

        if current_votes.sum() > best_votes.sum():
            best_model = current_model
            best_votes = current_votes
            logging.info("Current best model has {} votes at iteration {}".format(
                current_votes.sum(), ransac_iter))

    return best_model

def vis_model(image, model, show=True):
    import matplotlib.pyplot as plt
    edgelets = compute_edgelets(image)
    locations, directions, strengths = edgelets
    inliers = compute_votes(edgelets, model, 10) > 0

    edgelets = (locations[inliers], directions[inliers], strengths[inliers])
    locations, directions, strengths = edgelets
    vis_edgelets(image, edgelets, False)
    vp = model / model[2]
    plt.imshow(image)
    plt.plot(vp[0], vp[1], 'bo')
    # for i in range(locations.shape[0]):
    #     xax = [locations[i, 0], vp[0]]
    #     yax = [locations[i, 1], vp[1]]
    #     plt.plot(xax, yax, 'b-.')

    if show:
        plt.show()


def edgelet_lines(edgelets):

    locations, directions, _ = edgelets
    normals = np.zeros_like(directions)
    normals[:, 0] = directions[:, 1]
    normals[:, 1] = -directions[:, 0]
    p = -np.sum(locations * normals, axis=1)
    lines = np.concatenate((normals, p[:, np.newaxis]), axis=1)
    return lines

def compute_votes(edgelets, model, threshold_inlier=5):

    vp = model[:2] / model[2]

    locations, directions, strengths = edgelets

    est_directions = locations - vp
    dot_prod = np.sum(est_directions * directions, axis=1)
    abs_prod = np.linalg.norm(directions, axis=1) * \
        np.linalg.norm(est_directions, axis=1)
    abs_prod[abs_prod == 0] = 1e-5

    cosine_theta = dot_prod / abs_prod
    theta = np.arccos(np.abs(cosine_theta))

    theta_thresh = threshold_inlier * np.pi / 180
    return (theta < theta_thresh) * strengths


def reestimate_model(model, edgelets, threshold_reestimate=5):

    locations, directions, strengths = edgelets

    inliers = compute_votes(edgelets, model, threshold_reestimate) > 0
    locations = locations[inliers]
    directions = directions[inliers]
    strengths = strengths[inliers]

    lines = edgelet_lines((locations, directions, strengths))

    a = lines[:, :2]
    b = -lines[:, 2]
    est_model = np.linalg.lstsq(a, b)[0]
    return np.concatenate((est_model, [1.]))

def nullspace(A, atol=1e-13, rtol=0):
    A = np.atleast_2d(A)
    u, s, vh = np.linalg.svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns


def visualize(index, visual):
    image_one_list = ["1_001", "1_002", "1_003", "1_004", "1_005"];
    image_two_list = ["2_001", "2_002", "2_003"];
    image_three_list = ["3_001", "3_002"];
    image_a_list = ["a1", "a2", "a3", "a4", "a5", "a6"];
    img_path = "../images/input/"  + image_a_list[0] + ".jpg";
    image = plt.imread(img_path);
    if (visual):
        img_plot = plt.imshow(image);
        plt.show();
    return image;


def cv2_visualize(index, visual):
    image_one_list = ["1_001", "1_002", "1_003", "1_004", "1_005"];
    image_two_list = ["2_001", "2_002", "2_003"];
    image_three_list = ["3_001", "3_002"];
    image_a_list = ["a1", "a2", "a3", "a4", "a5", "a6"];
    img_path = "../images/input/"  + image_a_list[0] + ".jpg";
    img1 = cv2.imread(img_path, 0)
    if (visual):
        cv2.imshow('Image', img1);
    return img1;


def ransacF(pts1, pts2, M):
    print(np.shape(pts1))
    # print(pts2)
    final_F= 0
    max_inliers= 0
    num_iter= 1500
    tol= 0.0009
    num_inliers= 0
    N = pts1.shape[0]
    ones = np.ones([N,1])
    pts1_in= pts1
    pts2_in= pts2
    print("The shape of pts1_in is ", np.shape(pts1_in))

    pts1_org=  np.concatenate((pts1, ones), axis=1)
    pts2_org=  np.concatenate((pts2, ones), axis=1)

    print("The shape of pts1_org is ", np.shape(pts1_org))


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


# We will use seven point algorithm in this section
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
