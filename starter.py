#!/usr/bin/env python

import cv2
import numpy as np
import sys
import os

def fixKeyCode(code):
	# need this to fix our opencv bug
    return np.uint8(code).view(np.int8)

def labelAndWaitForKey(image, text):

    # Get the image height - the first element of its shape tuple.
    h = image.shape[0]

    display = image.copy()


    text_pos = (16, h-16)                # (x, y) location of text
    font_face = cv2.FONT_HERSHEY_SIMPLEX # identifier of font to use
    font_size = 1.0                      # scale factor for text
    
    bg_color = (0, 0, 0)       # RGB color for black
    bg_size = 3                # background is bigger than foreground
    
    fg_color = (255, 255, 255) # RGB color for white
    fg_size = 1                # foreground is smaller than background

    line_style = cv2.LINE_AA   # make nice anti-aliased text

    # Draw background text (outline)
    cv2.putText(display, text, text_pos,
                font_face, font_size,
                bg_color, bg_size, line_style)

    # Draw foreground text (middle)
    cv2.putText(display, text, text_pos,
                font_face, font_size,
                fg_color, fg_size, line_style)

    cv2.imshow('Image', display)

    # We could just call cv2.waitKey() instead of using a while loop
    # here, however, on some platforms, cv2.waitKey() doesn't let
    # Ctrl+C interrupt programs. This is a workaround.
    while fixKeyCode(cv2.waitKey(15)) < 0: pass

# Get command line arguments or print usage and exit
if len(sys.argv) > 2:
    proj_file = sys.argv[1]
    cam_file = sys.argv[2]
else:
    progname = os.path.basename(sys.argv[0])
    print >> sys.stderr, 'usage: '+progname+' PROJIMAGE CAMIMAGE'
    sys.exit(1)

# Load in our images as grayscale (1 channel) images
proj_image = cv2.imread(proj_file, cv2.IMREAD_GRAYSCALE)
cam_image = cv2.imread(cam_file, cv2.IMREAD_GRAYSCALE)

# Make sure they are the same size.
assert(proj_image.shape == cam_image.shape)

# Set up parameters for stereo matching (see OpenCV docs at
# http://goo.gl/U5iW51 for details).
min_disparity = 0
max_disparity = 16
window_size = 11
param_P1 = 0
param_P2 = 20000

# Create a stereo matcher object
matcher = cv2.StereoSGBM_create(min_disparity, 
                                max_disparity, 
                                window_size, 
                                param_P1, 
                                param_P2)

# Compute a disparity image. The actual disparity image is in
# fixed-point format and needs to be divided by 16 to convert to
# actual disparities.
disparity = matcher.compute(cam_image, proj_image) / 16.0

# Pop up the disparity image.
cv2.imshow('Disparity', disparity/disparity.max())
while fixKeyCode(cv2.waitKey(5)) < 0: pass

# Set up our intrinsic parameter matrix
#
#     [ f 0 u_0]
# K = [ 0 f v_0]
#     [ 0 0  1 ]

# Parameters:
f = 600
u_0 = 320
v_0 = 240

# Matrix:
K = f * np.eye(3)
K[2,2] = 1
K[0,2] = u_0
K[1,2] = v_0

baseline = 0.05 #in meters
b = np.array([baseline, 0, 0])

# Recall 
# [u; v; 1] = K [X; Y; Z]
# or q = K [X; Y; Z]
# Mapping each point $$q$$ through K^-1 will return a point P which is proportional to X, Y, Z
# The Z coordiante of each point P can be obtained by examining the disparity value 
# delta at each (u, v) location via:
# 
#         Z = (b * f) / delta
#
Zmax = 8 # meters

Z = np.zeros(disparity.size)
Z = (baseline * f) / disparity.flatten()
print(len(Z))

