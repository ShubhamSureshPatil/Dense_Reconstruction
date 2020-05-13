import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from utils import *
import cv2

def explore_match(win, img1, img2, kp_pairs, status = None, H = None):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1+w2), np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1+w2] = img2
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    if H is not None:
        corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        corners = np.int32( cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0) )
        cv2.polylines(vis, [corners], True, (255, 255, 255))

    if status is None:
        status = np.ones(len(kp_pairs), np.bool_)
    p1 = np.int32([kpp[0].pt for kpp in kp_pairs])
    p2 = np.int32([kpp[1].pt for kpp in kp_pairs]) + (w1, 0)

    green = (0, 255, 0)
    red = (0, 0, 255)
    white = (255, 255, 255)
    kp_color = (51, 103, 236)
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            col = green
            cv2.circle(vis, (x1, y1), 2, col, -1)
            cv2.circle(vis, (x2, y2), 2, col, -1)
        else:
            col = red
            r = 2
            thickness = 3
            cv2.line(vis, (x1-r, y1-r), (x1+r, y1+r), col, thickness)
            cv2.line(vis, (x1-r, y1+r), (x1+r, y1-r), col, thickness)
            cv2.line(vis, (x2-r, y2-r), (x2+r, y2+r), col, thickness)
            cv2.line(vis, (x2-r, y2+r), (x2+r, y2-r), col, thickness)
    vis0 = vis.copy()
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            cv2.line(vis, (x1, y1), (x2, y2), green)
    cv2.imshow(win, vis)   
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def filter_matches(kp1, kp2, matches, ratio = 0.75):
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append( kp1[m.queryIdx] )
            mkp2.append( kp2[m.trainIdx] )
    kp_pairs = zip(mkp1, mkp2)
    return kp_pairs

def match_images(img1, img2):
    """Given two images, returns the matches"""
    detector = cv2.xfeatures2d.SURF_create()
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    kp1, desc1 = detector.detectAndCompute(img1, None)
    kp2, desc2 = detector.detectAndCompute(img2, None)

    new_matches = matcher.match(desc1,desc2)

    raw_matches = matcher.knnMatch(desc1, trainDescriptors = desc2, k = 2) 
    kp_pairs = filter_matches(kp1, kp2, raw_matches)
    return kp_pairs, new_matches, kp1, kp2

def draw_matches(window_name, kp_pairs, img1, img2):
    """Draws the matches for """
    mkp1, mkp2 = zip(*kp_pairs)

    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])

    if len(kp_pairs) >= 4:
        H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
    else:
        H, status = None, None
    if len(p1):
        explore_match(window_name, img1, img2, kp_pairs, status, H)

def visualize_keypoints(img1, img2, p1, p2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1+w2), np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1+w2] = img2
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    color = (255,0,0);
    for (x1, y1), (x2, y2) in zip(p1, p2):
        print(int(x1),int(y1), int(x2), int(y2));
        print("\n");
        # cv2.circle(vis, int(x1, y1), 2, color, 2)
        # cv2.circle(vis, int(x2, y2), 2, color, 2)
    vis0 = vis.copy()
    for (x1, y1), (x2, y2) in zip(p1, p2):
        cv2.line(vis, (int(x1), int(y1), (int(x2), int(y2))), color = (255, 0, 0))
    cv2.imshow(win, vis)   
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # load the mat file
    mat = scipy.io.loadmat('good_ransac.mat')
    k1 = mat['K1'];
    k2 = mat['K2'];

    # Load the images
    img1 = cv2_visualize(0, False);
    img2 = cv2_visualize(1, False);

    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)




    
    """
    # Uncomment this block to compute the points
    kp_pairs, matches, kp1, kp2 = match_images(img1, img2);
    # print(matches)

    pts1_unhomo = [kp1[m.queryIdx].pt for m in matches];
    pts2_unhomo = [kp2[m.trainIdx].pt for m in matches];


    np.savez('matches.npz', p1 = pts1_unhomo, p2 = pts2_unhomo)

    data = np.load('matches.npz')
    p1 = data['p1'];
    p2 = data['p2'];

    # Visualizer is currently screwed
    # visualize_keypoints(img1, img2, p1, p2);
    p1_list = []
    p2_list = []

    for (x,y), (x2,y2) in zip(p1,p2):
        p1_list.append((x,y,1))
        p2_list.append((x2,y2,1))

    p1_arr = np.asarray(p1_list);
    p2_arr = np.asarray(p2_list);


    # print(np.shape(ones.T));
    # print(np.shape(p1_arr));
    # print(np.shape(p2_arr));
    # print(p1_arr)
    # print(p2_arr)


    im_width, im_height = np.shape(img1)
    M = max(im_height, im_width)

    F = ransacF(p1, p2, M);

    """







    

if __name__ == "__main__":
    main();
    