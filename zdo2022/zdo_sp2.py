import skimage.morphology
import numpy as np
import cv2 as cv
import random
import json
import sys
import os

from skimage.transform import (probabilistic_hough_line, hough_circle)
from skimage.feature import peak_local_max
from skimage.draw import circle_perimeter
from pathlib import Path

import podpurne_funkce

class InstrumentTracker2():
    def __init__(self):
        pass

    def create_annotation_output(self, filename, frame_id, object_id, x_px, y_px, annotation_timestamp):
        """
        :param filename: names of processed files
        :param frame_id: processed frames
        :param object_id: processed objects in order of listed points
        :param x_px: x coordinates of discovered points
        :param y_px: y coordinated of discovered points
        :param annotation_timestamp: time of annotation
        :return: annnotations
        """
        annotation={
            "filename": filename,
            "frame_id": frame_id,
            "object_id": object_id,
            "x_px": x_px,
            "y_px": y_px,
            "annotation_timestamp": annotation_timestamp,
        }

        return annotation

    def predict(self, path):
        """
        :param video_dirname: name of the directory containing the .png files the video consists of
        :return: annnotations
        """
        scale = 0.25

        filename = []
        frame_id = []
        object_id = []
        x_px = []
        y_px = []
        annotation_timestamp = []

        pth = Path(path)

        # Create video reader
        videoReader = cv.VideoCapture(path)

        if not videoReader.isOpened():
            print("Error opening the video file.")
            return

        # Get frame count
        frameCount = int(videoReader.get(cv.CAP_PROP_FRAME_COUNT))
        width = int(videoReader.get(cv.CAP_PROP_FRAME_WIDTH) * scale)
        height = int(videoReader.get(cv.CAP_PROP_FRAME_HEIGHT) * scale)
        size = (width, height)

        print("Frame count: ", frameCount)

        videoWriter = cv.VideoWriter("results/out2.avi", cv.VideoWriter_fourcc(*'DIVX'), 25, size);
        
        # Start reading frames from the video
        for i in range(0, frameCount):
            
            # Read a frame
            _, frame = videoReader.read()
            
            # Scale the frame down
            frame = podpurne_funkce.ScaleImage(frame, scale)
            
            hsv_image = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

            lower = np.array([0, 0, 0])
            upper = np.array([180, 60, 130])

            mask = cv.inRange(hsv_frame, lower, upper)
            hsv_frame[:,:,0] *= mask
            hsv_frame[:,:,1] *= mask
            hsv_frame[:,:,2] *= mask

            mask2 = hsv_frame[:,:,0] < (hsv_frame[:,:,1] * 0.9)
            hsv_frame[:,:,0] *= mask2
            hsv_frame[:,:,1] *= mask2
            hsv_frame[:,:,2] *= mask2

            edges = hsv_frame[:,:,0] + hsv_frame[:,:,1] + hsv_frame[:,:,2] > 0
            kernel = skimage.morphology.diamond(3).astype(np.uint8)
            edges = skimage.morphology.binary_closing(edges, kernel)

            lines = probabilistic_hough_line(edges, threshold=100, line_length=50, line_gap=2)

            #for line in lines:
                #p0, p1 = line
                #color = (0, 0, 255)
                #cv.line(hsv_frame, (p0[0], p0[1]), (p1[0], p1[1]), color)

            # Detect two radii
            hough_radii = np.arange(20, 100, 2)
            hough_res = hough_circle(edges, hough_radii)

            best_centers = []
            centers = []
            accums = []
            radii = []

            for radius, h in zip(hough_radii, hough_res):
                peaks = peak_local_max(h, num_peaks=2)
                centers.extend(peaks)
                accums.extend(h[peaks[:, 0], peaks[:, 1]])
                radii.extend([radius, radius])

            for idx in np.argsort(accums)[::-1][:10]:
                if (idx >= len(centers) or idx >= len(radii)):
                    break
                center_x, center_y = centers[idx]
                
                radius = radii[idx]
                cx, cy = circle_perimeter(center_y, center_x, radius)
                
                oob = cx >= 0
                cx *= oob
                oob = cx < frame.shape[1]
                cx *= oob

                oob = cy >= 0
                cy *= oob
                oob = cy < frame.shape[0]
                cy *= oob

                hsv_frame[cy, cx] = (255, 0, 0)
                
                best_centers.append(center_x)
                best_centers.append(center_y)

            #ball_positions = []
            #healths = []
            max_sat = 80

            for line in lines:
                ball_count = 1
                ball_health = np.zeros(ball_count)
                ball_health[:] = 500

                p0, p1 = line
                p0_min_dist = frame.shape[0] + frame.shape[1]
                p1_min_dist = p0_min_dist

                for center_pos in range(0, len(best_centers), 2):
                    a0 = ((p0[0] - best_centers[center_pos]), (p0[1] - best_centers[center_pos + 1]))
                    d0 = np.dot(a0, a0)
                    if (p0_min_dist > d0):
                        p0_min_dist = d0

                    a1 = ((p1[0] - best_centers[center_pos]), (p1[1] - best_centers[center_pos + 1]))
                    d1 = np.dot(a1, a1)
                    if (p1_min_dist > d1):
                        p1_min_dist = d1

                a = p0
                b = p1

                if (p0_min_dist > p1_min_dist):
                    a = p1
                    b = p0

                for ball in range(0, ball_count):
                    b_x = b[0]
                    b_y = b[1]

                    dir = (b[0] - a[0], b[1] - a[1])
                    l = np.linalg.norm(dir)
                    dir /= l

                    was_red = False

                    while (ball_health[ball] > 0):
                        ball_health[ball] -= 1
                        b_x += dir[0]
                        b_y += dir[1]

                        iy = int(b_y)
                        ix = int(b_x)

                        ix = max(0, ix)
                        iy = max(0, iy)

                        ix = min(frame.shape[1]-1, ix)
                        iy = min(frame.shape[0]-1, iy)

                        if hsv_image[iy][ix][0] < 15 or hsv_image[iy][ix][0] > 345:
                            was_red = True
                            continue

                        if hsv_image[iy][ix][1] > max_sat:
                            ball_health[ball] -= hsv_image[iy][ix][1] - max_sat

                            #r = 3
                            #min_sat = max_sat
                            #nx = ix
                            #ny = iy
                            #for y in range(iy-r, iy+r+1):
                            #    for x in range(iy-r, iy+r+1):
                            #        if hsv_image[y, x, 1] < min_sat:
                            #            min_sat = hsv_image[y, x, 1]
                            #            nx = x
                            #            ny = y

                            #ix = nx
                            #iy = ny

                            #if hsv_image[iy, ix, 1] < max_sat:
                            #    b_x = ix
                            #    b_y = iy

                            continue

                        if (was_red):
                            ball_health[ball] = 0 
                            b_x -= dir[0]
                            b_y -= dir[0]

                    filename.append(pth.parts[-1])
                    frame_id.append(i)
                    object_id.append(0)
                    x_px.append(ix * 4)
                    y_px.append(iy * 4)
                    annotation_timestamp.append(0)

                    cv.circle(img=frame, center = (ix, iy), radius = 3, color = (0, ball_health[ball], 0), thickness=-1)

            #cv.imshow('frame', hsv_frame)
            #cv.imwrite("results/" + ("%06d.PNG" % i), hsv_frame)
            #k = cv.waitKey(30) & 0xff
            videoWriter.write(frame)
            print("Writing frame " + str(i))

        # Release video reader
        videoReader.release()
        videoWriter.release()
        cv.destroyAllWindows()

        annotation = self.create_annotation_output(filename, frame_id, object_id, x_px, y_px, annotation_timestamp)
        
        with open("results/out2.json", "w") as output:
            json.dump(annotation, output, indent = 4)

def main():
    outDir = "results/"
    if not os.path.isdir(outDir):
        os.mkdir(outDir)

    path = sys.argv[1]
    tracker = InstrumentTracker2()
    tracker.predict(path)

if __name__ == "__main__":
    main()