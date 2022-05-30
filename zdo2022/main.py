import json
import sys
import os.path
import cv2 as cv
import numpy as np
import skimage
import skimage.color
import matplotlib.pyplot as plt
import math
from os import path

from skimage.transform import (hough_line, hough_line_peaks)
from podpurne_funkce import (GetImageFiles, GetFrame)
from filtering import (Filter, FilterKmeans)
from pathlib import Path

# Mark object at position in frame - cross
# Returns edited frame 
def MarkObjects(frame, position):
    newImage = frame.copy()

    for j in range(-10, 10):
        x = (int)(position[1][0])
        y = (int)(position[1][1])

        newImage[x+j,y] = (0, 0, 0)
        newImage[x,y+j] = (0, 0, 0)
        newImage[x+j+1,y+1] = (0, 0, 0)
        newImage[x+1,y+j+1] = (0, 0, 0)
        newImage[x+j-1,y-1] = (0, 0, 0)
        newImage[x-1,y+j-1] = (0, 0, 0)

    return newImage

# Mark object at position in frame - black square
# Returns edited frame 
def MarkObjects2(frame, position):
    newImage = frame.copy()

    width = int(frame.shape[0]) - 1
    height = int(frame.shape[1]) - 1

    for i in range(0, (int)(len(position[1])/2)):
        x = (int)(position[1][i*2 + 0])
        y = (int)(position[1][i*2 + 1])

        minX = max(x-5, 0)
        maxX = min(x+5, width)
    
        minY = max(y-5, 0)
        maxY = min(y+5, height)

        newImage[minX:maxX, minY:maxY] = (0, 0, 0)

    return newImage

# Logarithmic distance from one color to the other
def DistanceFromColor(color1, color2):
    c1_hsv = skimage.color.rgb2hsv(color1.reshape(1,1,3))
    c2_hsv = skimage.color.rgb2hsv(color2.reshape(1,1,3))
    dist = np.linalg.norm(c1_hsv - c2_hsv)
    return dist

# Get objects and  their properties from b&w image
def GetProperties(img):
    # Labels
    imlabel1 = skimage.measure.label(img, background=0)
    labs = np.unique(imlabel1)

    # Get descriptors   
    props = skimage.measure.regionprops(imlabel1, intensity_image=img)
    return props

# Find areas that can contain stationary objects. Using hough transformation
# Returns mask with said areas
def DetectStationary(frame):
    # New mask
    f2 = np.zeros(frame.shape)

    im = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    im = skimage.feature.canny(im, sigma=1)
    rows, cols = im.shape

    # Find 8 lines
    h, theta, d = hough_line(im)
    lhspace, langles, ldists = hough_line_peaks(h, theta, d, num_peaks=8)

    for rho, angle, dist in zip(lhspace, langles, ldists): 
        y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
        y1 = (dist - cols * np.cos(angle)) / np.sin(angle)

        x0 = (dist - 0 * np.sin(angle)) / np.cos(angle)
        x1 = (dist - rows * np.sin(angle)) / np.cos(angle)

        posS = [0, 0]
        posE = [cols, rows]
        if ((y0 > 0) and (y0 < rows)):
            posS[1] = y0

        if ((y1 > 0) and (y1 < rows)):
            posE[1] = y1

        if ((x0 > 0) and (x0 < cols)):
            posS[0] = x0

        if ((x1 > 0) and (x1 < cols)):
            posE[0] = x1

        # Draw line to img
        f2 = cv.line(f2, ((int)(posS[0]), (int)(posS[1])), ((int)(posE[0]), (int)(posE[1])), (255, 255, 255), 1)

    # Return mask
    return f2

# Interpolates positions using linear interpolation
# Returns list of edited positions
def InterpolatePositions(positions):
    newPositions = []

    # Adds empty frames at the start if neccessary
    for i in range(0, positions[0][0]):
        newPos = [i, positions[0][1]]
        newPositions.append(newPos)

    newPositions.append(positions[0])

    for i in range(1, len(positions)):
        closest = []
        
        # frame index - get 2 consecutive saved frames
        index2 = positions[i][0]
        index1 = positions[i-1][0]

        # distance in between the two frames time 
        diff = index2 - index1
        if (diff == 1):
            newPositions.append(positions[i])
            continue

        # find the closest position to each position from the second frame
        usedStarts = []
        minDist = float('inf')
        closestInd = -1
        for j in range(0, len(positions[i][1]), 2):
            minDist = float('inf')
            closestInd = -1
            for k in range(0, len(positions[i-1][1]), 2):
                if (usedStarts.count(k) > 0):
                    continue
                
                x1 = positions[i-1][1][k]
                y1 = positions[i-1][1][k+1]
                
                x2 = positions[i][1][j]
                y2 = positions[i][1][j+1]
                
                dist = math.sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2))
                if(dist < minDist):
                    minDist = dist
                    closestInd = k
            
            if(minDist < 100):
                closestPair = [j,closestInd]
                closest.append(closestPair)
                usedStarts.append(closestInd)
            
        # Create new positions for missing frames
        for j in range(1, diff):
            newPos2 = []
            for k in range(0, len(closest)):
                # Compute step in x and y
                stepX = (positions[i][1][closest[k][0]] - positions[i-1][1][closest[k][1]]) / diff
                stepY = (positions[i][1][closest[k][0]+1] - positions[i-1][1][closest[k][1]+1]) / diff
            
                # Linear interpolation time
                position = [0, 0]
                position[0] = positions[i-1][1][closest[k][1]] + j * stepX
                position[1] = positions[i-1][1][closest[k][1]+1] + j * stepY
                newPos2.append(position[0])
                newPos2.append(position[1])
                
            newPos = [index1 + j, newPos2]
            newPositions.append(newPos)

        newPositions.append(positions[i])

    return newPositions

# Pads out missing frames
def InterpolationPadding(positions):
    newPositions = []

    # Adds empty frames at the start if neccessary
    for i in range(0, positions[0][0]):
        newPos = [i, positions[0][1]]
        newPositions.append(newPos)

    newPositions.append(positions[0])

    for i in range(1, len(positions)):
        # frame index - get 2 consecutive saved frames
        index2 = positions[i][0]
        index1 = positions[i-1][0]

        # distance in between the two frames time 
        diff = index2 - index1
        for c in range(0, diff-1):
            newPositions.append([index1 + (c+1), positions[i-1][1]])
        
        newPositions.append(positions[i])

    return newPositions

# Is a tool-tip located at pos? Detected through the amount of red pixels in neighborhood
def DetectSurroundings(frame, pos, h):
    height, width, layers = frame.shape
    count = 0
    for i in range(-h, h+1):
        if ((pos[0] + i < 0) or (pos[0] + i >= height)):
            continue
        for j in range (-h, h+1):
            if ((pos[1] + j < 0) or (pos[1] + j >= width)):
                continue

            dist = DistanceFromColor(frame[pos[0] + i, pos[1] + j], np.array([0.1, 0.1, 0.5]))
            distOrange = DistanceFromColor(frame[pos[0] + i, pos[1] + j], np.array([0.2, 0.5, 0.8]))

            if (dist <= 0.4):
                count += 2
            if (distOrange <= 0.5):
                count -= 1
                
    # More than 50% -> tool
    res = False
    if (count/((h*2+1)*(h*2+1)) > 0.5):
        res = True
    return res    

# Process one frame from video, frame - full colored frame, mask - movement mask
# Returns list with x and y positions of detected tools, or [-1, -1] if no tool was found
def ProcessFrame(mask, frame, filtering):
    # print("frame shape: ", frame.shape)
    positions = [-1, -1]

    # Get separated objects and their poperties from mask image
    props = GetProperties(mask)
    # print("Found objects ", len(props))
    
    pixbox = (0, 0, frame.shape[0], frame.shape[1])

    # Go through detected objects
    for i in range(0, len(props)): 
        object_number = i

        # If detected "object" is the whole image -> continue
        bbox = props[object_number].bbox
        if (bbox == pixbox):
            continue
        
        #if (props[object_number].convex_area < 50):
        #    continue

    	# Go through pixels object
        coords = props[object_number].coords
        for c in range(0, len(coords)):
            x = coords[c][0]
            y = coords[c][1]

            dist = DistanceFromColor(frame[x, y], np.array([0.1, 0.1, 0.5]))
            if (dist <= 0.4):
                if (DetectSurroundings(frame, [x, y], 1)):
                    if (positions[0] == -1):
                        positions = []
                    positions.append(x)
                    positions.append(y) 

    # Stationary objects
    lines = DetectStationary(frame)

    mask2 = cv.bitwise_not(mask)
    mask2 = mask2.astype('uint8')*255
    mask2 = cv.bitwise_and(lines, lines, mask=mask2)

    props = GetProperties(mask2)
    # print("Found bg objects ", len(props))

    # Go through pixels object
    coords = np.argwhere(mask2 == 255) # props[object_number].coords
    for c in range(0, len(coords)):
            x = coords[c][0]
            y = coords[c][1]

            dist = DistanceFromColor(frame[x, y], np.array([0.1, 0.1, 0.5]))
            if (dist <= 0.4):
                if (DetectSurroundings(frame, [x, y], 1)):
                    if (positions[0] == -1):
                        positions = []
                    positions.append(x)
                    positions.append(y)

    positions2 = [-1, -1]
    if (filtering == 1):
        positions2 = Filter(positions)
    elif (filtering == 2):
        positions2 = FilterKmeans(positions)

    # Print number of found tools
    print("Found tools: ", (int)(len(positions2)/2))

    # Return
    if (len(positions2) == 0):
        positions2 = [-1, -1]

    return positions2

# Process video using the MOG operator, interpolates between positions in frames and outputs a video
def ProcessVideo2(imageDir, imageFiles, filtering, interpolation, outpath): 

    filenames = []
    object_ids = []
    frame_ids = []
    x_px = []
    y_px = []
    pth = Path(imageDir)

    print("Starting processing...")
    positions = []
    height, width, layers = (0,0,0)

    # Kernel and operator
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
    subtractor = cv.createBackgroundSubtractorMOG2()

    # Go through all frames
    for i in range(0, len(imageFiles)):
        filename = imageDir + imageFiles[i]
        
        # Get mask containing moving objects
        [frame, mask] = GetFrame(filename, kernel, subtractor)
        height, width, layers = frame.shape
        # mask = mask > 0.5

        # Process frame - get positions of moving tools from frame using the mask
        pos = ProcessFrame(mask, frame, filtering)

        # If tools detected, save positions
        if not (pos[0] == -1 and pos[1] == -1):
            positions.append([i, pos])

    
    # Interpolate missing positions
    if (interpolation == 1):
        positions = InterpolatePositions(positions)
    elif (interpolation == 2):
        positions = InterpolationPadding(positions) 

    print("p: ", len(positions), " f: ", len(imageFiles))

    # Output
    size = (width,height)
    out = cv.VideoWriter(outpath, cv.VideoWriter_fourcc(*'DIVX'), 1, size)

    # Draw in all frames a mark on detected tools
    for i in range(0, len(positions)):
        frameIndex = positions[i][0] # i
        filename = imageDir + imageFiles[frameIndex]
        [frame, mask] = GetFrame(filename, kernel, subtractor)

        for j in range(0, (int)(len(positions[i][1])/2)):
            x = (int)(positions[i][1][j*2 + 0])
            y = (int)(positions[i][1][j*2 + 1])

            filenames.append(pth.parts[-2])
            object_ids.append(0)
            frame_ids.append(frameIndex)
            x_px.append(y * 4)
            y_px.append(x * 4)

        outimg = MarkObjects2(frame, positions[i])
        out.write(outimg)
    
    out.release()

    annotation={
        "filename": filenames,
        "frame_id": frame_ids,
        "object_id": object_ids,
        "x_px": x_px,
        "y_px": y_px,
        "annotation_timestamp": [],
    }

    return annotation

class InstrumentTracker():
    def predict(self, dir, filtering, interpolation):
        if dir[-1] != "/":
            dir += "/"

        outDir = dir + "processed/"
        maskDir = outDir + "masks/"
        frameDir = outDir + "frames/"
        imageDir = dir + "images/"
        annotationFile = dir + "annotations.xml"
        outpath = "results/out.avi"

        #if not path.isdir(outDir):
            #os.mkdir(outDir)

        #if not path.isdir(maskDir):
            #os.mkdir(maskDir)

        #if not path.isdir(frameDir):
            #os.mkdir(frameDir)

        if not path.isdir("results"):
            os.mkdir("results")

        # Load file names
        imageFiles = GetImageFiles(imageDir)
        # Process files as video frames
        annotation = ProcessVideo2(imageDir, imageFiles, filtering, interpolation, outpath)
        
        with open("results/out.json", "w") as output:
            json.dump(annotation, output, indent = 4)

        return annotation

def main():

    dir = sys.argv[1]

    # 1 - default, 2 - kmeans
    filtering = int(sys.argv[2])

    # 1 - default, 2 - no interpolation 
    interpolation = int(sys.argv[3])

    InstrumentTracker().predict(dir, filtering, interpolation);

if __name__ == "__main__":
    main()