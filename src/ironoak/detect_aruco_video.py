import cv2
import depthai as dai
import pickle as pkl
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl

# from scipy.spatial.transform import Rotation as R
from cv2 import aruco # dont know why this is red underlined doesnet seem to have a problem
import time
import sys
from pathlib import Path


# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - two mono (grayscale) cameras
monoLeft = pipeline.createMonoCamera()
monoRight = pipeline.createMonoCamera()
stereo = pipeline.createStereoDepth()
spatialLocationCalculator = pipeline.createSpatialLocationCalculator()

xoutDepth = pipeline.createXLinkOut()
xoutSpatialData = pipeline.createXLinkOut()
xinSpatialCalcConfig = pipeline.createXLinkIn()
xoutRight = pipeline.createXLinkOut()

xoutDepth.setStreamName("depth")
xoutSpatialData.setStreamName("spatialData")
xinSpatialCalcConfig.setStreamName("spatialCalcConfig")
xoutRight.setStreamName('right')
#xoutRight.setStreamName('left')
# MonoCamera
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
monoRight.out.link(xoutRight.input)

outputDepth = True
outputRectified = False
lrcheck = False
subpixel = False

# StereoDepth
# stereo.setOutputDepth(outputDepth) # I commented both of these out bc it was complaining not sure if it is importnatn
# stereo.setOutputRectified(outputRectified)
stereo.initialConfig.setConfidenceThreshold(255)

stereo.setLeftRightCheck(lrcheck)
stereo.setSubpixel(subpixel)

monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

spatialLocationCalculator.passthroughDepth.link(xoutDepth.input)
stereo.depth.link(spatialLocationCalculator.inputDepth)

topLeft = dai.Point2f(0.4, 0.4)
bottomRight = dai.Point2f(0.6, 0.6)

spatialLocationCalculator.inputConfig.setWaitForMessage(False)
config = dai.SpatialLocationCalculatorConfigData()
config.depthThresholds.lowerThreshold = 100
config.depthThresholds.upperThreshold = 10000
config.roi = dai.Rect(topLeft, bottomRight)
spatialLocationCalculator.initialConfig.addROI(config)
spatialLocationCalculator.out.link(xoutSpatialData.input)
xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)

# Pipeline defined, now the device is assigned and pipeline is started
# device = dai.Device(pipeline) # clean up code adn maek it more usable
with dai.Device(pipeline) as device:

    # new: for calib data:
    calibFile = str((Path(__file__).parent / Path(f"calib_{device.getMxId()}.json")).resolve().absolute())
    if len(sys.argv) > 1:
        calibFile = sys.argv[1]

    calibData = device.readCalibration()
    calibData.eepromToJsonFile(calibFile)

    # M_left = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.LEFT, 1280, 720))
    # print("LEFT Camera resized intrinsics...")
    # print(M_left)
    # mtx = M_left #np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.LEFT, 1280, 720)) # new trying it
    #
    # D_left = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.LEFT))
    # print("LEFT Distortion Coefficients...")
    # [print(name + ": " + value) for (name, value) in
    #  zip(["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6", "s1", "s2", "s3", "s4", "τx", "τy"],
    #      [str(data) for data in D_left])]
    # dist = D_left# np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.LEFT)) # new trying it

    M_right = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT, 1280, 720))
    # print("RIGHT Camera resized intrinsics...")
    # print(M_right)
    mtx = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT, 1280, 720))

    D_right = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.RIGHT))
    # print("RIGHT Distortion Coefficients...")
    # [print(name + ": " + value) for (name, value) in
    #  zip(["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6", "s1", "s2", "s3", "s4", "τx", "τy"],
    #      [str(data) for data in D_right])]
    dist = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.RIGHT))

    # print(
    #     f"RGB FOV {calibData.getFov(dai.CameraBoardSocket.RGB)}, Mono FOV {calibData.getFov(dai.CameraBoardSocket.LEFT)}")

    # R1 = np.array(calibData.getStereoLeftRectificationRotation())
    # R2 = np.array(calibData.getStereoRightRectificationRotation())
    # M_right = np.array(calibData.getCameraIntrinsics(calibData.getStereoRightCameraId(), 1280, 720))

 #   H_left = np.matmul(np.matmul(M_right, R1), np.linalg.inv(M_left))
    # print("LEFT Camera stereo rectification matrix...")
    # print(H_left)

    #H_right = np.matmul(np.matmul(M_right, R1), np.linalg.inv(M_right))
    # print("RIGHT Camera stereo rectification matrix...")
    # print(H_right)

    #lr_extrinsics = np.array(calibData.getCameraExtrinsics(dai.CameraBoardSocket.LEFT, dai.CameraBoardSocket.RIGHT))
    # print("Transformation matrix of where left Camera is W.R.T right Camera's optical center")
    # print(lr_extrinsics)

    #l_rgb_extrinsics = np.array(calibData.getCameraExtrinsics(dai.CameraBoardSocket.LEFT, dai.CameraBoardSocket.RGB))
    # print("Transformation matrix of where left Camera is W.R.T RGB Camera's optical center")
    # print(l_rgb_extrinsics)

    # Output queue will be used to get the depth frames from the outputs defined above
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    spatialCalcQueue = device.getOutputQueue(name="spatialData", maxSize=4, blocking=False)
    spatialCalcConfigInQueue = device.getInputQueue("spatialCalcConfig")
    qRight = device.getOutputQueue(name="right", maxSize=4, blocking=False)
    color = (255, 255, 0)

    # ArUco declarations
    # mtx = np.load('/Users/Haviva/code/ArUco-marker-detection-with-DepthAi/datacalib_mtx_webcam.pkl', allow_pickle=True) # I will need to make changes to this
    # dist = np.load('/Users/Haviva/code/ArUco-marker-detection-with-DepthAi/datacalib_dist_webcam.pkl', allow_pickle=True) # This must be cameraMatrix and distCoeffs
    size_of_marker = 0.0145 #TODO adjust
    length_of_axis = 0.01
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250) # how much does this dictionary cover? Do i need to change this
    parameters = aruco.DetectorParameters_create() # make so you can use multiple dictionaries maybe ask user for data

    # New for fps counter:
    startTime = time.monotonic()
    counter = 0
    fps = 0

    while True:
        inDepth = depthQueue.get()  # blocking call, will wait until a new data has arrived
        inDepthAvg = spatialCalcQueue.get()  # blocking call, will wait until a new data has arrived
        inRight = qRight.tryGet()


        if inRight is not None:
            frameRight = inRight.getCvFrame()  # get mono right frame

        depthFrame = inDepth.getFrame()
        depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        depthFrameColor = cv2.equalizeHist(depthFrameColor)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

        # New for frame counter:
        counter += 1
        current_time = time.monotonic()
        if (current_time - startTime) > 1:
            fps = counter / (current_time - startTime)
            counter = 0
            startTime = current_time
            # print(fps)

        spatialData = inDepthAvg.getSpatialLocations()
        for depthData in spatialData:
            roi = depthData.config.roi
            roi = roi.denormalize(width=depthFrameColor.shape[1], height=depthFrameColor.shape[0])
            xmin = int(roi.topLeft().x)
            ymin = int(roi.topLeft().y)
            xmax = int(roi.bottomRight().x)
            ymax = int(roi.bottomRight().y)

            fontType = cv2.FONT_HERSHEY_TRIPLEX
            cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)
            cv2.putText(depthFrameColor, f"X: {int(depthData.spatialCoordinates.x)} mm", (xmin + 10, ymin + 20), fontType, 0.5, color)
            cv2.putText(depthFrameColor, f"Y: {int(depthData.spatialCoordinates.y)} mm", (xmin + 10, ymin + 35), fontType, 0.5, color)
            cv2.putText(depthFrameColor, f"Z: {int(depthData.spatialCoordinates.z)} mm", (xmin + 10, ymin + 50), fontType, 0.5, color)

        cv2.imshow("depth", depthFrameColor)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if frameRight is not None:
            # cv2.imshow("right", frameRight)
            # ArUco processing
            corners, ids, rejectedImgPoints = aruco.detectMarkers(frameRight, aruco_dict, parameters=parameters)
            frame_markers = aruco.drawDetectedMarkers(frameRight.copy(), corners, ids)
            for corner in corners:
                x_mid = (corner[0][1][0] + corner[0][3][0]) / 2
                y_mid = (corner[0][1][1] + corner[0][3][1]) / 2
                topLeft.x = (x_mid - 15) / 640
                topLeft.y = (y_mid - 15) / 400
                bottomRight.x = (x_mid + 15) / 640
                bottomRight.y = (y_mid + 15) / 400

            rvecs, tvecs, trash = aruco.estimatePoseSingleMarkers(corners, size_of_marker, mtx, dist)
            imaxis = aruco.drawDetectedMarkers(frameRight.copy(), corners, ids)
            imaxis = cv2.merge((imaxis,imaxis,imaxis)) # Just turn to color for easier display visability
            #print("tvecs.shape = {} rvecs.shape(squeeze) = {}\n{}".format(np.shape(tvecs), np.shape(np.squeeze(tvecs)), tvecs))
            if tvecs is not None:
                for i in range(len(tvecs)):
                    imaxis = cv2.drawFrameAxes(imaxis, mtx, dist, rvecs[i], tvecs[i], length_of_axis)  # this gets world coordinate system axis for object points # changed this and fixed it
                    rvec = np.squeeze(rvecs[0], axis=None)
                    tvec = np.squeeze(tvecs[0], axis=None)
                    tvec = np.expand_dims(tvec, axis=1)
                    rvec_matrix = cv2.Rodrigues(rvec)[0]
                    #print("rodriques\n",rvec_matrix)
                    #print("tvec:\n",tvec)
                    proj_matrix = np.hstack((rvec_matrix, tvec))
                    #print("proj matrix\n",proj_matrix)
                    euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
                    cv2.putText(imaxis, 'X: ' + str(int(euler_angles[0])), (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255))
                    cv2.putText(imaxis, 'Y: ' + str(int(euler_angles[1])), (115, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
                    cv2.putText(imaxis, 'Z: ' + str(int(euler_angles[2])), (200, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0))
            cv2.imshow('Aruco', imaxis)
            config.roi = dai.Rect(topLeft, bottomRight)
            cfg = dai.SpatialLocationCalculatorConfig()
            cfg.addROI(config)
            spatialCalcConfigInQueue.send(cfg)
