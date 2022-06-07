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
from zdo2022.podpurne_funkce import (ScaleImage, DistanceFromColor)
from zdo2022.interpolation import (InterpolatePositions, InterpolatePositionsK, InterpolationPadding)
from zdo2022.filtering import (Filter, FilterKmeans)
from pathlib import Path

# Find moving objects in image -> returns a mask
def GetMask(frame, kernel, subtractor):
    mask = subtractor.apply(frame)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    return mask

# Mark object at position in frame - square with color color
# Returns edited frame 
def MarkObjects(frame, position, color):
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

        newImage[minX:maxX, minY:maxY] = color

    return newImage

# Find areas that can contain objects using hough transformation.
# Returns mask with said areas.
def DetectStationary(frame):
    f2 = np.zeros(frame.shape)

    im = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    im = skimage.feature.canny(im, sigma=1, mode='reflect')
    rows, cols = im.shape

    # Find 8 lines
    hspace, angles, dists = hough_line(im)
    hspace, angles, dists = hough_line_peaks(hspace, angles, dists, num_peaks=8)

    for _, angle, dist in zip(hspace, angles, dists): 
        sina = np.sin(angle)
        cosa = np.cos(angle)
        
        x0 = -1
        x1 = -1
        y0 = -1
        y1 = -1

        if sina != 0:
            y0 = (dist) / sina
            y1 = (dist - cols * cosa) / sina

        if cosa != 0:
            x0 = (dist) / cosa
            x1 = (dist - rows * sina) / cosa

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

# Is a tool-tip located at pos? Detected through the amount of red pixels in neighborhood
def DetectSurroundings(frame, pos, h):
    height, width, _ = frame.shape
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


def AssignIDs(positionsK, width, height):
    # the one most centered - needleholder -> 0?
    # the one most on top - scissors -> 1?
    # the remaining one - tweezers -> 2?
    
    centerX = width / 2
    centerY = height / 2
    ids = []
    
    
    # go through all the frames:
    for i in range(0, len(positionsK)):
        needleHolderInd = -1
        scissorsInd = -1
        tweezersInd = -1
        
        minDist = float('inf')
        # go through all the positions in this frame (prolly 3):
        for j in range(0, len(positionsK[i][1]), 2):
            dist = math.sqrt((centerX - positionsK[i][1][j + 1])*(centerX - positionsK[i][1][j + 1]) + (centerY - positionsK[i][1][j])*(centerY - positionsK[i][1][j]))
            if (dist < minDist):
                minDist = dist
                needleHolderInd = j
                
                
        maxUp = float('inf')
        for j in range(0, len(positionsK[i][1]), 2):
            if(j == needleHolderInd):
                continue

            if(positionsK[i][1][j] < maxUp):
                maxUp = positionsK[i][1][j]
                scissorsInd = j
                
                
        for j in range(0, len(positionsK[i][1]), 2):
            if(j == needleHolderInd or j == scissorsInd):
                continue

            tweezersInd = j
            break
            
        for j in range(0, len(positionsK[i][1]), 2):
            if(j == needleHolderInd):
                ids.append(0)
            elif(j == scissorsInd):
                ids.append(1)
            elif(j == tweezersInd):
                ids.append(2)
        
            
    return ids

# Process one frame from video, frame - full colored frame, mask - movement mask
# Returns list with x and y positions of detected tools, or [-1, -1] if no tool was found
def ProcessFrame(frame):
    positions = [-1, -1]
    positionsK = [-1, -1]

    # Stationary detection
    mask2 = DetectStationary(frame)

    # Go through pixels object
    coords = np.argwhere(mask2 == 255)
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

    positions2 = Filter(positions)
    positionsK = positions
    if (len(positions) >= 3):
        positionsK = FilterKmeans(positions)

    # Print number of found tools
    print("Found tools: ", (int)(len(positions2)/2))

    # Return
    if (len(positions2) == 0):
        positions2 = [-1, -1]
        positionsK = [-1, -1]

    return positions2, positionsK

# Process video using the MOG operator, interpolates between positions in frames and outputs a video
def Process(path):

    # Initialize variables
    positionsK = []
    positions = []

    filename = []
    frame_id = []
    x_px = []
    y_px = []
    annotation_timestamp = []

    height = 0
    width = 0
    scale = 0.25
    skip = 25
    mul = 1/scale
    
    pth = Path(path)
    
    # Kernel and operator
    # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
    # subtractor = cv.createBackgroundSubtractorMOG2()

    # Create video reader
    videoReader = cv.VideoCapture(path)

    if not videoReader.isOpened():
        print("Error opening the video file.")
        return

    # Get frame count
    frameCount = int(videoReader.get(cv.CAP_PROP_FRAME_COUNT))
    print("Frame count: ", frameCount)
    frameCount = 75

    # Start reading frames from the video
    for i in range(0, frameCount):

        # Read a frame
        read, frame = videoReader.read()
        
        if i % skip != 0 and i != frameCount - 1:
            continue
        
        print("Processing frame ", i)
        
        if not read:
            print("Failed to process frame ", i)
            return

        # Scale the frame down
        frame = ScaleImage(frame, scale)

        # Save frame shape
        height, width, _ = frame.shape

        # Get frame and mask containing moving objects
        # mask = GetMask(frame, kernel, subtractor)
        #cv.imshow("Mask", frame)
        #cv.waitKey(50)

        # Process frame - get positions of moving tools from frame using the mask
        [pos, posK] = ProcessFrame(frame)

        # If tools detected, save positions
        if not (pos[0] == -1 and pos[1] == -1):
            positions.append([i, pos])
            positionsK.append([i, posK])

    # Release video reader
    videoReader.release()

    # Interpolate missing positions
    positions = InterpolatePositions(positions)
    positionsK = InterpolatePositionsK(positionsK)

    # Output
    size = (width,height)
    videoWriter = cv.VideoWriter("results/out.avi", cv.VideoWriter_fourcc(*'DIVX'), 25, size)
    
    # Ids are returned in the order they appear in the positionsK list
    ids = AssignIDs(positionsK, width, height) 
    #print(ids)
    
    # Draw in all frames a mark on detected tools
    videoReader = cv.VideoCapture(path)
    
    # Start reading frames from the video
    for i in range(0, frameCount):

        # Read a frame
        _, frame = videoReader.read()
        
        # Scale the frame down
        frame = ScaleImage(frame, scale)
        
        # Mark the tool positions
        outimg = MarkObjects(frame, positions[i], (0, 0, 0))
        outimg = MarkObjects(outimg, positionsK[i], (0, 0, 255))

        # Write output image        
        videoWriter.write(outimg)

        # Append data for each position
        for i in range(0, (int)(len(positionsK[i][1])/2)):
            
            x = (int)(positionsK[i][1][i*2 + 0])
            y = (int)(positionsK[i][1][i*2 + 1])
            
            filename.append(pth.parts[-1])
            frame_id.append(i)
            x_px.append(x * mul)
            y_px.append(y * mul)
    
    # Release video reader and video writer
    videoReader.release()
    videoWriter.release()

    #cv.destroyAllWindows()
    print("Done.")

    annotation={
        "filename": filename,
        "frame_id": frame_id,
        "object_id": ids,
        "x_px": x_px,
        "y_px": y_px,
        "annotation_timestamp": annotation_timestamp,
    }

    return annotation

class InstrumentTracker():
    def predict(self, path):
        annotation = Process(path)
        
        with open("results/out.json", "w") as output:
            json.dump(annotation, output, indent = 4)

        return annotation

def main():
    #print(skimage.__version__)
    InstrumentTracker().predict("9.mp4")

if __name__ == "__main__":
    main()