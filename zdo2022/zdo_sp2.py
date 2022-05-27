from __future__ import annotations
from configparser import RawConfigParser
from itertools import accumulate
import random
from podpurne_funkce import GetImageFiles
from podpurne_funkce import ScaleImage
from skimage.transform import (probabilistic_hough_line, hough_circle)
from skimage.feature import peak_local_max
from skimage.draw import circle_perimeter
from os import path

import numpy as np
import cv2 as cv
import os.path
import skimage.morphology


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

        print(annotation)

        return annotation

    def predict(self, video_dirname):
        """
        :param video_dirname: name of the directory containing the .png files the video consists of
        :return: annnotations
        """

        outDir = "results/"
        if not path.isdir(outDir):
            os.mkdir(outDir)

        filename = []
        frame_id = []
        object_id = []
        x_px = []
        y_px = []

        filename.append(video_dirname)

        image_files = GetImageFiles(video_dirname + "/images")
        #print(image_files)


        for i in range(0, len(image_files),1 ):
            image_filename = video_dirname + "/images/" + image_files[i]
            frame = cv.imread(image_filename)
            frame = ScaleImage(frame, 0.25)

            #image_filename = video_dirname + "/images/" + image_files[i+1]
            #frame2 = cv.imread(image_filename)
            #frame2 = scale_image(frame2, 0.25)

            #frame = cv.addWeighted(frame, 0.5, frame2, 0.5, 0.0)
            
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
                hsv_frame[cy, cx] = (255, 0, 0)
                
                best_centers.append(center_x)
                best_centers.append(center_y)

            #ball_positions = []
            #healths = []
            max_sat = 80;

            for line in lines:
                ball_count = 10
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
                    rr = random.uniform(-0.05, 0.05)
                    dir /= l
                    if (rr > 0):
                        dir *= -1
                    rx = random.uniform(-0.05, 0.05)
                    ry = random.uniform(-0.05, 0.05)
                    dir[0] += rx
                    dir[1] += ry
                    l = np.linalg.norm(dir)
                    dir /= l

                    was_red = False

                    while (ball_health[ball] > 0):
                        b_x += dir[0]
                        b_y += dir[1]

                        iy = int(b_y)
                        ix = int(b_x)

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

                        
                    cv.circle(img=frame, center = (ix, iy), radius = 3, color = (0, 0, 0), thickness=-1)

            cv.imshow('frame', hsv_frame)
            #cv.imwrite("results/" + ("%06d.PNG" % i), hsv_frame)
            k = cv.waitKey(30) & 0xff

        cv.destroyAllWindows()

        annotation_timestamp = [None] * len(filename)

        return self.create_annotation_output(filename, frame_id, object_id, x_px, y_px, annotation_timestamp)

def main():
    video_dirname = "D:/Data/ZDO/224"
    tracker = InstrumentTracker2()
    tracker.predict(video_dirname)

if __name__ == "__main__":
    main()