#!/usr/bin/env python

import cv2
import numpy as np
import sys
import os
import cmath

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
f = 600.0
u_0 = 320
v_0 = 240

# Matrix:
K = f * np.eye(3)
K[2,2] = 1
K[0,2] = u_0
K[1,2] = v_0

# Baseline is in meters
baseline = 0.05 
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


# We want to restrict the max depth 
# Matt suggests using Zmax = 8
# Let's use np.clip and pass in None for the minimum
# Actually, let's NOT use np.clip let's just make a mask
min_d = baseline * f / (Zmax)
mask = disparity > min_d

# Flatten our mask out so the dimensions work out
mask = mask.flatten()

# Initialize to the appropriate size
X = np.zeros(disparity.size)
Y = np.zeros(disparity.size)

# Let's calculate the pixel coordinates using np.arange
u_pixels = np.arange(cam_image.shape[1])
v_pixels = np.arange(cam_image.shape[0])

# Create a meshgrid for the helper function to check
up, vp = np.meshgrid(u_pixels, v_pixels)

# Calculate X and Y; NOTE: forgetting the Z still
X = (u_pixels - u_0) / f
Y = (v_pixels - v_0) / f

# Create the appropriate meshgrid
Xv, Yv = np.meshgrid(X, Y)

# Flatten them so they're just long vectors
X =  Xv.flatten()
Y =  Yv.flatten()

# Multiply the two vectors point wise so that the proportionality is scaled
X = Z * X
Y = Z * Y

def check_pixels(u_pix, v_pix, Xv, Yv, Zvals, u0, v0, f):
    '''
    Info:
        This helper function takes the meshgrid of X and Y BEFORE being multiplied by Z 
        and checks to make sure that the math was done correctly. 

        In other words, makes sure that:
            X = Z * (u - u0) / f
            Y = Z * (v - v0) / f
        
        Note, Z must be reshaped into a grid of the same size as Xv and Yv

        All three of Xv, Yv, and Z must be the same shape. 

        Returns: 
            TRUE  - if the calculation is done correctly and the transformations make sense
            FALSE - otherwise
    '''

    # Reshape our Z so that it'd 2D just like Xv and Yv
    Z = Zvals.reshape(Xv.shape)

    # Create the appropriate equations
    # Don't forget, we didn't multiply the left hand side by Z before passing that in so double check
    u_lhs = Z * Xv
    u_rhs = Z * (u_pix - u0)/f

    # Repeat the same for the Y's and v's 
    v_lhs = Z * Yv
    v_rhs = Z * (v_pix - v0)/f

    # Use pythonic check to make sure every value in our 2D array is the same
    # Note, we need to use np.isclose to make sure that the floats are actually equal

    if all(all(val for val in row) for row in np.isclose(u_lhs, u_rhs, rtol=1e-5)) and \
        all(all(val for val in row) for row in np.isclose(v_lhs, v_rhs, rtol=1e-5)):
        return True
    return False

# Let's make sure that everything matches and our equations work out
print("The transformation has been computed successfully? (T/F)")
print(check_pixels(up, vp, Xv, Yv, Z, u_0, v_0, f))

# Form one 3 x n matrix just to use available methods
# Not sure how to skip the transpose in the next line
# Don't forget about the mask
final = np.vstack((X[mask], Y[mask], Z[mask]))

# Transpose so it's in the right n x 3 format
final = final.T

print final.shape

# Note, in python 3 change raw_input --> input
fname = raw_input("What file would you like to save your transformation to?: ")
np.savez(fname, final)









