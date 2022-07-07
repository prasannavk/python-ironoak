#!/usr/bin/env python3

import math
import time

import cv2
import depthai as dai
import numpy as np
from calc import HostSpatialsCalc
from pupil_apriltags import Detector

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
monoLeft = pipeline.create(dai.node.MonoCamera)
aprilTag = pipeline.create(dai.node.AprilTag)
# From main_calc:
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)

xoutMono = pipeline.create(dai.node.XLinkOut)
xoutAprilTag = pipeline.create(dai.node.XLinkOut)

xoutMono.setStreamName("mono")
xoutAprilTag.setStreamName("aprilTagData")


# Properties
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)

aprilTag.initialConfig.setFamily(dai.AprilTagConfig.Family.TAG_36H11)

# from https://pypi.org/project/pupil-apriltags/
at_detector = Detector(families='tag36h11', nthreads=1, quad_decimate=1.0, quad_sigma=0.0, refine_edges=1, decode_sharpening=0.25, debug=0)

# From main_calc
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

stereo.initialConfig.setConfidenceThreshold(255)
stereo.setLeftRightCheck(False)  # possibly change this to False
stereo.setSubpixel(False)

# From main_calc Linking
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutDepth.setStreamName("depth")  # Do you use both of these evaluate?
stereo.depth.link(xoutDepth.input)

xoutDepth = pipeline.create(dai.node.XLinkOut)  # this seems repetative you can prob delete
xoutDepth.setStreamName("disp")
stereo.disparity.link(xoutDepth.input)

# Linking
aprilTag.passthroughInputImage.link(xoutMono.input)
monoLeft.out.link(aprilTag.inputImage)
aprilTag.out.link(xoutAprilTag.input)
# always take the latest frame as apriltag detections are slow
aprilTag.inputImage.setBlocking(False)
aprilTag.inputImage.setQueueSize(1)

# advanced settings, configurable at runtime
aprilTagConfig = aprilTag.initialConfig.get()
aprilTagConfig.quadDecimate = 4
aprilTagConfig.quadSigma = 0
aprilTagConfig.refineEdges = True
aprilTagConfig.decodeSharpening = 0.25
aprilTagConfig.maxHammingDistance = 1
aprilTagConfig.quadThresholds.minClusterPixels = 5
aprilTagConfig.quadThresholds.maxNmaxima = 10
aprilTagConfig.quadThresholds.criticalDegree = 10
aprilTagConfig.quadThresholds.maxLineFitMse = 10
aprilTagConfig.quadThresholds.minWhiteBlackDiff = 5
aprilTagConfig.quadThresholds.deglitch = False
aprilTag.initialConfig.set(aprilTagConfig)


# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queue will be used to get the mono frames from the outputs defined above
    monoQueue = device.getOutputQueue("mono", 8, False)
    aprilTagQueue = device.getOutputQueue("aprilTagData", 8, False)

    # From main_calc
    # Output queue will be used to get the depth frames from the outputs defined above
    depthQueue = device.getOutputQueue(name="depth")
    dispQ = device.getOutputQueue(name="disp")

    hostSpatials = HostSpatialsCalc(device)  # Calls the class on calc.py to perform depth calculations
    roi_radius = 10
    hostSpatials.setDeltaRoi(roi_radius)

    color = (0, 255, 0)

    startTime = time.monotonic()
    counter = 0
    fps = 0

    while True:
        inFrame = monoQueue.get()

        # from main_calc
        depthFrame = depthQueue.get().getFrame()  # numpy image of x,y,Z

        # Get disparity frame for nicer depth visualization
        depthFrameColor = dispQ.get().getFrame()
        depthFrameColor = (depthFrameColor * (255 / stereo.initialConfig.getMaxDisparity())).astype(np.uint8)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_JET)

        counter += 1
        current_time = time.monotonic()
        if (current_time - startTime) > 1:
            fps = counter / (current_time - startTime)
            counter = 0
            startTime = current_time
            print(fps)

        monoFrame = inFrame.getFrame()
        frame = cv2.cvtColor(monoFrame, cv2.COLOR_GRAY2BGR)

        # On host, pupil april tag detector
        grayframe = frame[:, :, 0]
        tags = at_detector.detect(grayframe, estimate_tag_pose=False, camera_params=None, tag_size=None)
        i = 0
        for at in tags:
            topLeft = at.corners[0]
            bottomLeft = at.corners[1]
            bottomRight = at.corners[2]
            topRight = at.corners[3]

            grayframe = frame

            fontType = cv2.FONT_HERSHEY_TRIPLEX

            center = (int((topLeft[0] + bottomRight[0]) / 2), int((topLeft[1] + bottomRight[1]) / 2))
            cv2.circle(grayframe, center, 5, (0, 0, 255), -1)

            # This draws lines around the apriltag in the apriltag frame
            cv2.line(grayframe, (int(topLeft[0]), int(topLeft[1])), (int(topRight[0]), int(topRight[1])), color, 2, cv2.LINE_AA, 0)
            cv2.line(grayframe, (int(topRight[0]), int(topRight[1])), (int(bottomRight[0]), int(bottomRight[1])), color, 2, cv2.LINE_AA, 0)
            cv2.line(
                grayframe, (int(bottomRight[0]), int(bottomRight[1])), (int(bottomLeft[0]), int(bottomLeft[1])), color, 2, cv2.LINE_AA, 0
            )
            cv2.line(grayframe, (int(bottomLeft[0]), int(bottomLeft[1])), (int(topLeft[0]), int(topLeft[1])), color, 2, cv2.LINE_AA, 0)

            idStr = "ID: " + str(aprilTag.id)
            cv2.putText(grayframe, idStr, center, fontType, 0.5, color)

            # grabs the x and y points for the center
            x = center[0]
            y = center[1]

            spatials, centroid = hostSpatials.calc_spatials(depthFrame, center)  # centroid == x/y in our case

            # This draws on the apriltag frame the depth calculations and frames the roi
            cv2.rectangle(grayframe, (x - roi_radius, y - roi_radius), (x + roi_radius, y + roi_radius), (255, 255, 255), 1)
            cv2.putText(
                grayframe,
                "X: " + ("{:.1f}m".format(spatials['x'] / 1000) if not math.isnan(spatials['x']) else "--"),
                (x + 10, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.putText(
                grayframe,
                "Y: " + ("{:.1f}m".format(spatials['y'] / 1000) if not math.isnan(spatials['y']) else "--"),
                (x + 10, y + 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.putText(
                grayframe,
                "Z: " + ("{:.1f}m".format(spatials['z'] / 1000) if not math.isnan(spatials['z']) else "--"),
                (x + 10, y + 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # This draws the box around the apriltag in the disparity map
            cv2.circle(depthFrameColor, center, 5, (0, 0, 255), -1)
            cv2.line(depthFrameColor, (int(topLeft[0]), int(topLeft[1])), (int(topRight[0]), int(topRight[1])), color, 2, cv2.LINE_AA, 0)
            cv2.line(
                depthFrameColor, (int(topRight[0]), int(topRight[1])), (int(bottomRight[0]), int(bottomRight[1])), color, 2, cv2.LINE_AA, 0
            )
            cv2.line(
                depthFrameColor,
                (int(bottomRight[0]), int(bottomRight[1])),
                (int(bottomLeft[0]), int(bottomLeft[1])),
                color,
                2,
                cv2.LINE_AA,
                0,
            )
            cv2.line(
                depthFrameColor, (int(bottomLeft[0]), int(bottomLeft[1])), (int(topLeft[0]), int(topLeft[1])), color, 2, cv2.LINE_AA, 0
            )

            cv2.putText(depthFrameColor, idStr, center, fontType, 0.5, color)

            # This draws on the disparity map the depth calculations in roi
            cv2.rectangle(depthFrameColor, (x - roi_radius, y - roi_radius), (x + roi_radius, y + roi_radius), (255, 255, 255), 1)
            cv2.putText(
                depthFrameColor,
                "X: " + ("{:.1f}m".format(spatials['x'] / 1000) if not math.isnan(spatials['x']) else "--"),
                (x + 10, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.putText(
                depthFrameColor,
                "Y: " + ("{:.1f}m".format(spatials['y'] / 1000) if not math.isnan(spatials['y']) else "--"),
                (x + 10, y + 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.putText(
                depthFrameColor,
                "Z: " + ("{:.1f}m".format(spatials['z'] / 1000) if not math.isnan(spatials['z']) else "--"),
                (x + 10, y + 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Show the frame for apriltag
        cv2.imshow("mono", grayframe)

        # Show the frame for disparity
        cv2.imshow("depth", depthFrameColor)

        if cv2.waitKey(1) == ord('q'):
            break
