#!/usr/bin/env python3

import cv2
import depthai as dai
import math
from calc import HostSpatialsCalc
from utility import *
import numpy as np
import time

# Note: anywhere I commented the word new below that is what i copied from spatial_location_calculator from depthai

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
aprilTag = pipeline.create(dai.node.AprilTag)
manip = pipeline.create(dai.node.ImageManip)

# new also from main_calc
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)

xoutAprilTag = pipeline.create(dai.node.XLinkOut)
xoutAprilTagImage = pipeline.create(dai.node.XLinkOut)

xoutAprilTag.setStreamName("aprilTagData")
xoutAprilTagImage.setStreamName("aprilTagImage")

# Properties
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)

manip.initialConfig.setResize(480, 270)
manip.initialConfig.setFrameType(dai.ImgFrame.Type.GRAY8)

aprilTag.initialConfig.setFamily(dai.AprilTagConfig.Family.TAG_36H11)

## From main_calc
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

stereo.initialConfig.setConfidenceThreshold(255)
stereo.setLeftRightCheck(True)
stereo.setSubpixel(False)

# Linking
aprilTag.passthroughInputImage.link(xoutAprilTagImage.input)
camRgb.video.link(manip.inputImage)
manip.out.link(aprilTag.inputImage)
aprilTag.out.link(xoutAprilTag.input)
# always take the latest frame as apriltag detections are slow
aprilTag.inputImage.setBlocking(False)
aprilTag.inputImage.setQueueSize(1)

## From main_calc
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutDepth.setStreamName("depth")
stereo.depth.link(xoutDepth.input)

xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutDepth.setStreamName("disp")
stereo.disparity.link(xoutDepth.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    # Output queue will be used to get the mono frames from the outputs defined above
    manipQueue = device.getOutputQueue("aprilTagImage", 8, False)
    aprilTagQueue = device.getOutputQueue("aprilTagData", 8, False)

    ## From main_calc
    # Output queue will be used to get the depth frames from the outputs defined above
    depthQueue = device.getOutputQueue(name="depth")
    dispQ = device.getOutputQueue(name="disp")
    text = TextHelper()
    hostSpatials = HostSpatialsCalc(device) # Conversion to depth functions class
    roi_radius = 10
    hostSpatials.setDeltaRoi(roi_radius)

    color = (0, 255, 0)

    startTime = time.monotonic()
    counter = 0
    fps = 0

    while (True):
        inFrame = manipQueue.get()

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

        monoFrame = inFrame.getFrame()
        frame = cv2.cvtColor(monoFrame, cv2.COLOR_GRAY2BGR)
        aprilTagData = aprilTagQueue.get().aprilTags
        for aprilTag in aprilTagData:
            topLeft = aprilTag.topLeft
            topRight = aprilTag.topRight
            bottomRight = aprilTag.bottomRight
            bottomLeft = aprilTag.bottomLeft
            # print("topLeft is {}, {}".format(topLeft.x, topLeft.y))
            # print("topRight is {}, {}".format(topRight.x, topRight.y))
            # print("bottomRight is {}, {}".format(bottomRight.x, bottomRight.y))
            # print("bottomLeft is {}, {}".format(bottomLeft.x, bottomLeft.y))

            center = (int((topLeft.x + bottomRight.x) / 2), int((topLeft.y + bottomRight.y) / 2))
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

            cv2.line(frame, (int(topLeft.x), int(topLeft.y)), (int(topRight.x), int(topRight.y)), color, 2,
                     cv2.LINE_AA, 0)
            cv2.line(frame, (int(topRight.x), int(topRight.y)), (int(bottomRight.x), int(bottomRight.y)), color, 2,
                     cv2.LINE_AA, 0)
            cv2.line(frame, (int(bottomRight.x), int(bottomRight.y)), (int(bottomLeft.x), int(bottomLeft.y)), color,
                     2, cv2.LINE_AA, 0)
            cv2.line(frame, (int(bottomLeft.x), int(bottomLeft.y)), (int(topLeft.x), int(topLeft.y)), color, 2,
                     cv2.LINE_AA, 0)

            idStr = "ID: " + str(aprilTag.id)
            cv2.putText(frame, idStr, center, cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)

            ## DEAL WITH DEPTH
            x = center[0]
            y = center[1]
            spatials, centroid = hostSpatials.calc_spatials(depthFrame, center)  # centroid == x/y in our case

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


        cv2.putText(frame, "Fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4,
                    (255, 255, 255))

        cv2.imshow("April tag frame", frame)

        # Show the frame
        cv2.imshow("depth", depthFrameColor)

        if cv2.waitKey(1) == ord('q'):
            break

