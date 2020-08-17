# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 10:15:12 2019

@author: Uma
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label
import matplotlib.image as mpimg
# Read in cars and notcars  
vehicle = glob.glob('C:\\Users\\Uma\\Anacondanew\\envs\\opencv-env\\vehicles\\images\\*.*')
nonvehicle=glob.glob("C:/Users/Uma/Anacondanew/envs/opencv-env/ML proj/non-vehicles/images/*.*")
# Do some data exploration
rand_vehicle_ind = np.random.randint(0, len(vehicle))
rand_nonvehicle_ind = np.random.randint(0, len(nonvehicle))
print("Number of vehicle", len(vehicle))
print("Number of nonvehicle", len(nonvehicle))
print("vehicle image shape", mpimg.imread(vehicle[rand_vehicle_ind]).shape)
print("Nonvehicle image shape", mpimg.imread(nonvehicle[rand_nonvehicle_ind]).shape)
# Visualize an example car and noncar image
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,10))
ax1.imshow(cv2.cvtColor(cv2.imread(vehicle[rand_vehicle_ind]), cv2.COLOR_BGR2RGB))
ax1.axis('off')
ax1.set_title("vehicle image")
ax2.imshow(cv2.cvtColor(cv2.imread(nonvehicle[rand_nonvehicle_ind]), cv2.COLOR_BGR2RGB))
ax2.axis('off')
ax2.set_title("Non vehicle image")
#Functions to extract HOG, Spatial Binning, Color Histogram features
# Define a function to extract features from a list of images
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = cv2.imread(file)
        # apply color conversion if other than 'RGB'. I use BGR as I read images in OpenCV
        if color_space == 'RGB':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif color_space == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            
        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features) 

            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features
# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualize=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=True, 
                       visualize=vis, feature_vector=feature_vec)
        return features
    # Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features
# Define a function to compute color histogram features 
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Couple of useful functions mainly used for plotting    
def bgr2rgb(img) :
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def rgb2bgr(img) :
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#Feature Extraction and Optimum Classifier
 # Read in a random image
ind = np.random.randint(0, len(vehicle))
image = cv2.imread(vehicle[ind])
# Define HOG parameters
orient = 11
pix_per_cell = 8
cell_per_block = 2
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
##
##Extract HOG visualizations for H, S, and V channels
hog_channel = 0 # H in HSV color space
features, hog_image_h = get_hog_features(hsv_image[:,:,hog_channel], orient, 
                       pix_per_cell, cell_per_block, 
                        vis=True, feature_vec=False)

hog_channel = 1 # S in HSV color space
features, hog_image_s = get_hog_features(hsv_image[:,:,hog_channel], orient, 
                      pix_per_cell, cell_per_block, 
                      vis=True, feature_vec=False)

hog_channel = 2 # V in HSV color space
features, hog_image_v = get_hog_features(hsv_image[:,:,hog_channel], orient, 
                    pix_per_cell, cell_per_block, 
                        vis=True, feature_vec=False)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4,figsize=(15,15))
ax1.imshow(bgr2rgb(image))
ax1.axis('off')

ax1.set_title("Original car image")
ax2.imshow(hog_image_h, cmap = 'gray')
ax2.axis('off')
ax2.set_title("H channel HOG")
ax3.imshow(hog_image_s, cmap = 'gray')
ax3.axis('off')
ax3.set_title("S channel HOG")
ax4.imshow(hog_image_v, cmap = 'gray')
ax4.axis('off')
ax4.set_title("V channel HOG")
# Check for the best classifier with the following six orientation options; I keep the other HOG parameters fixed
color_space = 'HLS'
orient = 11
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"

spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off

vehicle_features = extract_features(vehicle, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
nonvehicle_features = extract_features(nonvehicle, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)

X = np.vstack((vehicle_features, nonvehicle_features)).astype(np.float64) 
#Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)
#Define the labels vector
y = np.hstack((np.ones(len(vehicle_features)), np.zeros(len(nonvehicle_features))))
from sklearn.svm import SVC
#Split up data into randomized training (80%) and test (20%) sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)
print('Feature vector length:', len(X_train[0]))
# Read in a random image
ind = np.random.randint(0, len(vehicle))
image = cv2.imread(vehicle[ind])
#car_ind = np.random.randint(0, len(car_images))
# Plot an example of raw and scaled features
fig = plt.figure(figsize=(12,4))
plt.subplot(131)
plt.imshow(image)
plt.title('Original Image')
plt.subplot(132)
plt.plot(X[ind])
plt.title('Raw Features')
plt.subplot(133)
plt.plot(scaled_X[ind])
plt.title('Normalized Features')
fig.tight_layout()
# Use a linear SVC 
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
print("HLS color space, 11 orientations, Spatial binning and Color histograms")
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
print("----------------------------")
print('Training Accuracy of SVC = ', round(svc.score(X_train, y_train), 4))
print("----------------------------")

t=time.time()

## Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy
    
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
#    
    # Array of bounding boxes where cars are detected
    bbox_list = []
    
    #draw_img = np.copy(img)
    #img = img.astype(np.float32)/255
    img_tosearch = img[ystart:ystop,:,:]
    
    ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HLS) 
    

    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)-1
    nyblocks = (ch1.shape[0] // pix_per_cell)-1 
    nfeat_per_block = orient*cell_per_block**2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1 
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
#    
#    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell
#
#            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
#            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)
#            print("No of spatial features are "+ str(len(spatial_features)))
#            print("No of hist features are "+ str(len(hist_features)))
#
#            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))       
            test_prediction = svc.predict(test_features)          
#            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                #cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6)
                bbox_list.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
                
    return bbox_list
# Search for cars in four regions within a new image. No need to search at the top of the image (0 to 336) as there are only 
# buildings, bill boards, and trees
image = cv2.imread('test1.jpg')
draw_img = np.copy(image)
bbox_list = []

ystart = 400
ystop = 528
scale = 1.0
bbox_list.append(find_cars(draw_img, ystart, ystop, scale, svc, X_scaler, orient, 
                           pix_per_cell, cell_per_block, spatial_size, hist_bins))
draw_img = draw_boxes(image, bbox_list[0], color=(0, 0, 255), thick=6)

ystart = 400
ystop = 592
scale = 1.5
bbox_list.append(find_cars(draw_img, ystart, ystop, scale, svc, X_scaler, orient, 
                           pix_per_cell, cell_per_block, spatial_size, hist_bins))
draw_img = draw_boxes(draw_img, bbox_list[1], color=(0, 255, 0), thick=6)

ystart = 400
ystop = 656
scale = 2.0
bbox_list.append(find_cars(draw_img, ystart, ystop, scale, svc, X_scaler, orient, 
                           pix_per_cell, cell_per_block, spatial_size, hist_bins))
draw_img = draw_boxes(draw_img, bbox_list[2], color=(255, 0, 0), thick=6)

ystart = 336
ystop = 656
scale = 2.5
bbox_list.append(find_cars(draw_img, ystart, ystop, scale, svc, X_scaler, orient, 
                           pix_per_cell, cell_per_block, spatial_size, hist_bins))
draw_img = draw_boxes(draw_img, bbox_list[3], color=(255, 255, 0), thick=6)

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(15,10))
ax1.imshow(bgr2rgb(image))
ax1.axis('off')
ax1.set_title("Original image")
ax2.imshow(bgr2rgb(draw_img))
ax2.axis('off')
ax2.set_title("Bounding boxes")

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,255,0), 6)
    # Return the image
    return img
image = cv2.imread('test1.jpg')
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4,figsize=(15,15))
ax1.imshow(bgr2rgb(image))
ax1.axis('off')
ax1.set_title("Original image")

draw_img = np.copy(image)
bbox_list = []

ystart = 400
ystop = 528
scale = 1.0
bbox_list.append(find_cars(draw_img, ystart, ystop, scale, svc, X_scaler, orient, 
                           pix_per_cell, cell_per_block, spatial_size, hist_bins))
draw_img = draw_boxes(image, bbox_list[0], color=(0, 0, 255), thick=6)

ystart = 400
ystop = 592
scale = 1.5
bbox_list.append(find_cars(draw_img, ystart, ystop, scale, svc, X_scaler, orient, 
                           pix_per_cell, cell_per_block, spatial_size, hist_bins))
draw_img = draw_boxes(draw_img, bbox_list[1], color=(0, 255, 0), thick=6)

ystart = 400
ystop = 656
scale = 2.0
bbox_list.append(find_cars(draw_img, ystart, ystop, scale, svc, X_scaler, orient, 
                           pix_per_cell, cell_per_block, spatial_size, hist_bins))
draw_img = draw_boxes(draw_img, bbox_list[2], color=(255, 0, 0), thick=6)

ystart = 336
ystop = 656
scale = 2.5
bbox_list.append(find_cars(draw_img, ystart, ystop, scale, svc, X_scaler, orient, 
                           pix_per_cell, cell_per_block, spatial_size, hist_bins))
draw_img = draw_boxes(draw_img, bbox_list[3], color=(255, 255, 0), thick=6)

# Plot image with bounding boxes
ax2.imshow(bgr2rgb(draw_img))
ax2.axis('off')
ax2.set_title("With Bounding boxes")

heat = np.zeros_like(image[:,:,0]).astype(np.float)

# Add heat to each box in box list
for box in bbox_list:
    heat = add_heat(heat, box)

# Apply threshold to help remove false positives
heat = apply_threshold(heat, 2)

# Visualize the heatmap when displaying    
heatmap = np.clip(heat, 0, 255)

# Find final boxes from heatmap using label function
labels = label(heatmap)

draw_img = draw_labeled_bboxes(bgr2rgb(np.copy(image)), labels)

ax3.imshow(heatmap, cmap='hot')
ax3.axis('off')
ax3.set_title("Heat Map")

ax4.imshow(draw_img)
ax4.axis('off')
ax4.set_title("vehicle Detection")