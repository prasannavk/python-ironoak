import sys
import time
from pathlib import Path
import argparse

import cv2
import depthai as dai
import numpy as np
from cv2 import aruco  # dont know why this is red underlined doesnet seem to have a problem

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
# ap.add_argument("-o", "--output", required=True,
# 	help="path to output image containing ArUCo tag")
# ap.add_argument("-i", "--id", type=int, required=True,
# 	help="ID of ArUCo tag to generate")
ap.add_argument("-t", "--type", type=str,
	default="DICT_ARUCO_ORIGINAL",
	help="type of ArUCo tag to generate such as: DICT_ARUCO_ORIGINAL (default)")

ap.add_argument("-s","--size_marker", type=float,
                default=0.0145,
                help="Size of Aruco marker in meters (0.0145 Default)")
ap.add_argument("-l","--length_axis", type=float,
                default=0.01,
                help="Length of axis (0.01 Default)")
args = vars(ap.parse_args())

# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - two mono (grayscale) cameras
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
spatialLocationCalculator = pipeline.create(dai.node.SpatialLocationCalculator)

xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutSpatialData = pipeline.create(dai.node.XLinkOut)
xinSpatialCalcConfig = pipeline.create(dai.node.XLinkIn)
xoutRight = pipeline.create(dai.node.XLinkOut)

xoutDepth.setStreamName("depth")
xoutSpatialData.setStreamName("spatialData")
xinSpatialCalcConfig.setStreamName("spatialCalcConfig")
xoutRight.setStreamName('right')
# xoutRight.setStreamName('left')

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
# topLeft = dai.Point2f(0.0, 0.0)  # I changed this so it doesnt start getting depth until after aruco board is detected
# bottomRight = dai.Point2f(0.0, 0.0)

spatialLocationCalculator.inputConfig.setWaitForMessage(False)  # changed from false to true
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

    # Output queue will be used to get the depth frames from the outputs defined above
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    spatialCalcQueue = device.getOutputQueue(name="spatialData", maxSize=4, blocking=False)
    spatialCalcConfigInQueue = device.getInputQueue("spatialCalcConfig")
    qRight = device.getOutputQueue(name="right", maxSize=4, blocking=False)
    color = (255, 255, 0)

    # ArUco declarations
    # mtx_where = '/Users/Haviva/code/ArUco-marker-detection-with-DepthAi/datacalib_mtx_webcam.pkl'
    # mtx = np.load(mtx_where, allow_pickle=True)
    # dist_where = '/Users/Haviva/code/ArUco-marker-detection-with-DepthAi/datacalib_dist_webcam.pkl'
    # dist = np.load(dist_where, allow_pickle=True)

    # aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)  # TODO expand this so multiple dicts are covered
    # new define names of each possible ArUco tag OpenCV supports
    ARUCO_DICT = {
        "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
        "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
        "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
        "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
        "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
        "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
        "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
        "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
        "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
        "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
        "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
        "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
        "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
        "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
        "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
        "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
        "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
        "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
        "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
        "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
        "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
    }
    # verify that the supplied ArUCo tag exists and is supported by
    # OpenCV
    if ARUCO_DICT.get(args["type"], None) is None:
        print("[INFO] ArUCo tag of '{}' is not supported".format(
            args["type"]))
        sys.exit(0)
    # load the ArUCo dictionary
    aruco_dict = aruco.Dictionary_get(ARUCO_DICT[args["type"]])
    parameters = aruco.DetectorParameters_create()  # make so you can use multiple dictionaries maybe ask user for data
    # load size parameters for ArUCo
    size_of_marker = args["size_marker"]
    length_of_axis = args["length_axis"]

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
            xmin = int(roi.topLeft().x) # seems to do self.x1 here for class
            ymin = int(roi.topLeft().y)
            xmax = int(roi.bottomRight().x)
            ymax = int(roi.bottomRight().y)


            # maybe make a function for this below
            fontType = cv2.FONT_HERSHEY_TRIPLEX
            cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)
            cv2.putText(depthFrameColor, f"X: {int(depthData.spatialCoordinates.x)} mm", (xmin + 10, ymin + 20), fontType, 0.5, color)
            cv2.putText(depthFrameColor, f"Y: {int(depthData.spatialCoordinates.y)} mm", (xmin + 10, ymin + 35), fontType, 0.5, color)
            cv2.putText(depthFrameColor, f"Z: {int(depthData.spatialCoordinates.z)} mm", (xmin + 10, ymin + 50), fontType, 0.5, color)

            # Experiment:
            color_n = (255, 54, 51)
            cv2.rectangle(depthFrameColor, (xmin, int(ymin + 67.04)), (xmax, int(ymax + 67.04)), color_n, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)
            cv2.putText(depthFrameColor, f"X: {int(depthData.spatialCoordinates.x)} mm", (xmin + 10, int(ymin + 87.04)),
                        fontType, 0.5, color_n)
            cv2.putText(depthFrameColor, f"Y: {int(depthData.spatialCoordinates.y)} mm", (xmin + 10, int(ymin + 102.04)),
                        fontType, 0.5, color_n)
            cv2.putText(depthFrameColor, f"Z: {int(depthData.spatialCoordinates.z)} mm", (xmin + 10, int(ymin + 117.04)),
                        fontType, 0.5, color_n)
            # center = (int((topLeft[0] + bottomRight[0]) / 2), int((topLeft[1] + bottomRight[1]) / 2))
            cv2.circle(depthFrameColor, (int(depthData.spatialCoordinates.x), int(depthData.spatialCoordinates.y + 67.04)), 5, color_n, -1)

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
            imaxis = cv2.merge((imaxis, imaxis, imaxis))  # Just turn to color for easier display visability
            # print("tvecs.shape = {} rvecs.shape(squeeze) = {}\n{}".format(np.shape(tvecs), /n
            # np.shape(np.squeeze(tvecs)), tvecs))
            if tvecs is not None:
                for i in range(len(tvecs)):
                    imaxis = cv2.drawFrameAxes(imaxis, mtx, dist, rvecs[i], tvecs[i], length_of_axis)
                    rvec = np.squeeze(rvecs[0], axis=None)
                    tvec = np.squeeze(tvecs[0], axis=None)
                    tvec = np.expand_dims(tvec, axis=1)
                    rvec_matrix = cv2.Rodrigues(rvec)[0]
                    # print("rodriques\n",rvec_matrix)
                    # print("tvec:\n",tvec)
                    print("rvec:\n{}\ntvec:\n{}. . .".format(rvec_matrix, tvec))
                    #rvec_matrix = np.identity(3)
                    proj_matrix = np.hstack((rvec_matrix, tvec))

                    proj_matrix_inv = np.hstack((np.linalg.inv(rvec_matrix),-tvec))
                    world_point = np.array([depthData.spatialCoordinates.x,depthData.spatialCoordinates.y,depthData.spatialCoordinates.z,1000.0])/1000.0
                    #world_pointT = np.transpose(world_point)
                    proj_wpinv = np.matmul(proj_matrix_inv,world_point)
                    proj_wp = np.matmul(proj_matrix,world_point)
                    print("PT from:\n{}\nPT to inv:\n{}\nPt to orig:\n{}\n______".format(world_point,proj_wpinv,proj_wp))

                    # print("proj matrix\n",proj_matrix)
                    euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
                    cv2.putText(imaxis, 'X: ' + str(int(euler_angles[0])), (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255))
                    cv2.putText(imaxis, 'Y: ' + str(int(euler_angles[1])), (115, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
                    cv2.putText(imaxis, 'Z: ' + str(int(euler_angles[2])), (200, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0))
            cv2.imshow('Aruco', imaxis)

            config.roi = dai.Rect(topLeft, bottomRight)
            cfg = dai.SpatialLocationCalculatorConfig()
            cfg.addROI(config)
            spatialCalcConfigInQueue.send(cfg)
