from skimage import feature, color, transform, io
import matplotlib.pyplot as plt
import scipy
import sympy
import numpy as np
import logging
import sys, os
from utils import *
import sys
import warnings


# Ignore all the warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore");


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

def question1():
    # Load the images
    image_one_list = ["1_001", "1_002", "1_003", "1_004", "1_005"];
    image_two_list = ["2_001", "2_002", "2_003"];
    image_three_list = ["3_001", "3_002"];
    image_a_list = ["a1", "a2", "a3", "a4", "a5", "a6"];

    # To visualize the Image
    image = visualize(0, False);

    edgelets1 = compute_edgelets(image);

    # frist vanishing points
    vp1 = ransac_vanishing_point(edgelets1, 2000, threshold_inlier=5);

    # second vanshing points
    edgelets2 = remove_inliers(vp1, edgelets1, 10);
    vp2 = ransac_vanishing_point(edgelets2, num_ransac_iter=2000,
                                threshold_inlier=5);
    vp2 = reestimate_model(vp2, edgelets2, threshold_reestimate=5);

    # third vanishing points
    edgelets3 = remove_inliers(vp2, edgelets2, 10);
    vp3 = ransac_vanishing_point(edgelets3, num_ransac_iter=2000,
                                threshold_inlier=5);

    # visualizing edgelets
    # vis_edgelets(image, edgelets1, col= 'g-')
    # vis_edgelets(image, edgelets2, col= 'r-')
    # vis_edgelets(image, edgelets3, col= 'y-')
    print ("Number of edgelets detected are: \nEdgelets 1: {} \nEdgelets 2: {} \nEdgelets 3: {}".
        format(edgelets1[0].shape[0], edgelets2[0].shape[0], 
        edgelets3[0].shape[0]))
        

    # Uncomment to Visualize vanishing points
    # vis_model(image, vp1);
    # vis_model(image, vp2);
    # vis_model(image, vp3);

def question2():
    print("In Question 2")
    image_one_list = ["1_001", "1_002", "1_003", "1_004", "1_005"];
    image_two_list = ["2_001", "2_002", "2_003"];
    image_three_list = ["3_001", "3_002"];
    image_a_list = ["a1", "a2", "a3", "a4", "a5", "a6"];

    # To visualize the Image
    image = visualize(0, False);

    edgelets1 = compute_edgelets(image);

    # frist vanishing points
    vp1 = ransac_vanishing_point(edgelets1, 2000, threshold_inlier=5);

    # second vanshing points
    edgelets2 = remove_inliers(vp1, edgelets1, 10);
    vp2 = ransac_vanishing_point(edgelets2, num_ransac_iter=2000,
                                threshold_inlier=5);
    vp2 = reestimate_model(vp2, edgelets2, threshold_reestimate=5);

    # third vanishing points
    edgelets3 = remove_inliers(vp2, edgelets2, 10);
    vp3 = ransac_vanishing_point(edgelets3, num_ransac_iter=2000,
                                threshold_inlier=5);


    # Normalize the vanishing points
    vx1, vy1 = vp1[0]/vp1[2], vp1/vp1[2];
    vx2, vy2 = vp2[0]/vp2[2], vp2/vp2[2];
    vx3, vy3 = vp3[0]/vp3[2], vp3/vp3[2];

    # print(vx1, vx2, vx3);
    # print(vy1, vy2, vy3);
    p11 = vx1 * vx2 + vy1 + vy2, vx1 + vx2;
    p21 = vx1 + vx2;
    p31 = vy1 + vy2;

    p12 = vx3 * vx2 + vy3 + vy2;
    p22 = vx3 + vx2;    
    p32 = vy3 + vy2;

    p13 = vx3 * vx1 + vy3 + vy1;
    p23 = vx3 + vx1;    
    p33 = vy3 + vy1;

    van_matrix = np.matrix([[p11, p21, p31, 1], [p12, p22, p32, 1], [p13, p23, p33, 1]])

    # van_maxtrix = np.matrix([[vx1 * vx2 + vy1 + vy2, vx1 + vx2, vy1 + vy2, 1], [vx3 * vx2 + vy3 + vy2, vx3 + vx2, vy3 + vy2, 1],
    # [vx3 * vx1 + vy3 + vy1, vx3 + vx1, vy3 + vy1, 1]
    # ], dtype = float)
    
    # Find the nullspace of van_matrix
    # w = nullspace(van_maxtrix);
    w = np.linalg.svd(van_matrix, full_matrices=True);
    print(np.shape(van_matrix));
    # print(w);




if __name__ == "__main__":
    # question1();
    question2();
