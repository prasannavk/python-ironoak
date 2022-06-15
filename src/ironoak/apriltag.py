#!/usr/bin/env python3

import cv2
import depthai as dai
import time
from calc import HostSpatialsCalc
from utility import *
import numpy as np
import math


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

# From main_calc
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

stereo.initialConfig.setConfidenceThreshold(255)
stereo.setLeftRightCheck(True)
stereo.setSubpixel(False)

# From main_calc Linking
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutDepth.setStreamName("depth")
stereo.depth.link(xoutDepth.input)

xoutDepth = pipeline.create(dai.node.XLinkOut)
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

    text = TextHelper()
    hostSpatials = HostSpatialsCalc(device)
    roi_radius = 10
    hostSpatials.setDeltaRoi(roi_radius)

    color = (0, 255, 0)

    startTime = time.monotonic()
    counter = 0
    fps = 0

    while(True):
        inFrame = monoQueue.get()

        # from main_calc
        depthFrame = depthQueue.get().getFrame()  # numpy image of x,y,Z

        # Get disparity frame for nicer depth visualization
        depthFrameColor = dispQ.get().getFrame()
        depthFrameColor = (depthFrameColor * (255 / stereo.initialConfig.getMaxDisparity())).astype(np.uint8)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_JET)

        counter+=1
        current_time = time.monotonic()
        if (current_time - startTime) > 1 :
            fps = counter / (current_time - startTime)
            counter = 0
            startTime = current_time

        monoFrame = inFrame.getFrame()
        frame = cv2.cvtColor(monoFrame, cv2.COLOR_GRAY2BGR)

        aprilTagData = aprilTagQueue.get().aprilTags
        for aprilTag in aprilTagData:
            topLeft = aprilTag.topLeft
            topRight = aprilTag.topRight
            bottomRight = aprilTag.bottomRight
            bottomLeft = aprilTag.bottomLeft

            center = (int((topLeft.x + bottomRight.x) / 2), int((topLeft.y + bottomRight.y) / 2))
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

            cv2.line(frame, (int(topLeft.x), int(topLeft.y)), (int(topRight.x), int(topRight.y)), color, 2, cv2.LINE_AA, 0)
            cv2.line(frame, (int(topRight.x), int(topRight.y)), (int(bottomRight.x), int(bottomRight.y)), color, 2, cv2.LINE_AA, 0)
            cv2.line(frame, (int(bottomRight.x), int(bottomRight.y)), (int(bottomLeft.x), int(bottomLeft.y)), color, 2, cv2.LINE_AA, 0)
            cv2.line(frame, (int(bottomLeft.x), int(bottomLeft.y)), (int(topLeft.x), int(topLeft.y)), color, 2, cv2.LINE_AA, 0)

            idStr = "ID: " + str(aprilTag.id)
            cv2.putText(frame, idStr, center, cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)

            ## DEAL WITH DEPTH
            x = center[0]
            y = center[1]
            print(x, y)
            spatials, centroid = hostSpatials.calc_spatials(depthFrame, center)  # centroid == x/y in our case
            cv2.circle(depthFrameColor, center, 5, (0, 0, 255), -1)

            text.rectangle(depthFrameColor, (x - roi_radius, y - roi_radius), (x + roi_radius, y + roi_radius))
            text.putText(depthFrameColor,
                         "X: " + ("{:.1f}m".format(spatials['x'] / 1000) if not math.isnan(spatials['x']) else "--"),
                         (x + 10, y + 20))
            text.putText(depthFrameColor,
                         "Y: " + ("{:.1f}m".format(spatials['y'] / 1000) if not math.isnan(spatials['y']) else "--"),
                         (x + 10, y + 35))
            text.putText(depthFrameColor,
                         "Z: " + ("{:.1f}m".format(spatials['z'] / 1000) if not math.isnan(spatials['z']) else "--"),
                         (x + 10, y + 50))
            cv2.line(depthFrameColor, (int(topLeft.x), int(topLeft.y)), (int(topRight.x), int(topRight.y)), color, 2, cv2.LINE_AA,
                     0)
            cv2.line(depthFrameColor, (int(topRight.x), int(topRight.y)), (int(bottomRight.x), int(bottomRight.y)), color, 2,
                     cv2.LINE_AA, 0)
            cv2.line(depthFrameColor, (int(bottomRight.x), int(bottomRight.y)), (int(bottomLeft.x), int(bottomLeft.y)), color, 2,
                     cv2.LINE_AA, 0)
            cv2.line(depthFrameColor, (int(bottomLeft.x), int(bottomLeft.y)), (int(topLeft.x), int(topLeft.y)), color, 2,
                     cv2.LINE_AA, 0)
            cv2.putText(depthFrameColor, idStr, center, cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)

            # For apriltag
            text.rectangle(frame, (x - roi_radius, y - roi_radius), (x + roi_radius, y + roi_radius))
            text.putText(frame,
                         "X: " + ("{:.1f}m".format(spatials['x'] / 1000) if not math.isnan(spatials['x']) else "--"),
                         (x + 10, y + 20))
            text.putText(frame,
                         "Y: " + ("{:.1f}m".format(spatials['y'] / 1000) if not math.isnan(spatials['y']) else "--"),
                         (x + 10, y + 35))
            text.putText(frame,
                         "Z: " + ("{:.1f}m".format(spatials['z'] / 1000) if not math.isnan(spatials['z']) else "--"),
                         (x + 10, y + 50))

            cv2.putText(frame, "Fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255,255,255))

            cv2.imshow("mono", frame)

            # Show the frame
            cv2.imshow("depth", depthFrameColor)



        if cv2.waitKey(1) == ord('q'):
            break


