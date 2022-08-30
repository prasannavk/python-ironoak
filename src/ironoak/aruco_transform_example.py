import sys
import time
from pathlib import Path
import argparse

import cv2
import depthai as dai
import numpy as np
from cv2 import aruco  # dont know why this is red underlined doesnet seem to have a problem


def mycomposeRt(R,t):
    '''
    Compose 3x3 R and 3x1 t into a 4x4 transfrom T
    Args:
        R (): 3x3 numpy array
        t (): 3x1 numpy array
    Returns:
        4x4 T(R,t)
    '''
    T = np.zeros((4, 4), dtype=float)
    T[0:3,0:3] = R
    T[0:3,3] = t
    T[3,3] =  1.0
    return T

def myinvertT(T):
    '''
    Invert 4x4 numpy transform matrix T
    Args:
        T (): 4x4 float numpy Transform matrix
    Returns: T^-1
    '''
    return np.linalg.inv(T)

def my3to4pt(p):
    '''
    Convert a 3x1 numpy 3d point to homogeneous 4x1 vector
    Args:
        p (): 3x1 numpy 3D point
    Returns: p(x,y,z) -> p(x,y,z,1)
    '''
    four1s = np.ones(4, dtype=float)
    four1s[0:3] = p
    return four1s





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

# MonoCamera THE_ 400 480 720 800 _P
resolution = dai.MonoCameraProperties.SensorResolution.THE_800_P
monoLeft.setResolution(resolution)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(resolution)
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

topLeft = [dai.Point2f(0.4, 0.4)]
bottomRight = [dai.Point2f(0.6, 0.6)]
# topLeft = dai.Point2f(0.0, 0.0)  # I changed this so it doesnt start getting depth until after aruco board is detected
# bottomRight = dai.Point2f(0.0, 0.0)

spatialLocationCalculator.inputConfig.setWaitForMessage(False)  # changed from false to true
config = dai.SpatialLocationCalculatorConfigData()
config.depthThresholds.lowerThreshold = 100
config.depthThresholds.upperThreshold = 10000
config.roi = dai.Rect(topLeft[0], bottomRight[0])
spatialLocationCalculator.initialConfig.addROI(config)
spatialLocationCalculator.out.link(xoutSpatialData.input)
xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)

# Pipeline defined, now the device is assigned and pipeline is started
# device = dai.Device(pipeline) # clean up code adn maek it more usable
foo = 0
with dai.Device(pipeline) as device:

    # new: for calib data:
    calibFile = str((Path(__file__).parent / Path(f"calib_{device.getMxId()}.json")).resolve().absolute())
    if len(sys.argv) > 1:
        calibFile = sys.argv[1]

    calibData = device.readCalibration()
    calibData.eepromToJsonFile(calibFile)

    #M_right = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT, 1280, 720))
    # print("RIGHT Camera resized intrinsics...")
    # print(M_right)
    mtx = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT, 1280, 720))

    # D_right = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.RIGHT))
    # print("RIGHT Distortion Coefficients...")
    # [print(name + ": " + value) for (name, value) in
    #  zip(["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6", "s1", "s2", "s3", "s4", "τx", "τy"],
    #      [str(data) for data in D_right])]
    distortion = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.RIGHT))

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

    # aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
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

    # Image sizes
    mono_height, mono_width, mono_channels = 0, 0, 1
    depth_height, depth_width, depth_channels = 0, 0, 1

    # New for fps counter:
    startTime = time.monotonic()
    counter = 0
    fps = 0
    # Collect and process images
    while True:
        inDepth = depthQueue.get()  # blocking call, will wait until a new data has arrived
        inDepthAvg = spatialCalcQueue.get()  # blocking call, will wait until a new data has arrived
        inRight = qRight.tryGet()

        if inRight is not None:
            frameRight = inRight.getCvFrame()  # get mono right frame
            if mono_height == 0:
                mono_height, mono_width, *_ = frameRight.shape

        depthFrame = inDepth.getFrame()
        if depth_height == 0:
            depth_height, depth_width, *_ = depthFrame.shape
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
            cv2.putText(depthFrameColor, f"X: {int(depthData.spatialCoordinates.x)} mm", (xmin + 35, ymin + 25), fontType, 0.5, color)
            cv2.putText(depthFrameColor, f"Y: {int(depthData.spatialCoordinates.y)} mm", (xmin + 35, ymin + 40), fontType, 0.5, color)
            cv2.putText(depthFrameColor, f"Z: {int(depthData.spatialCoordinates.z)} mm", (xmin + 35, ymin + 55), fontType, 0.5, color)

        cv2.imshow("depth", depthFrameColor)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if frameRight is not None:
            # cv2.imshow("right", frameRight)
            # ArUco processing
            corners, ids, rejectedImgPoints = aruco.detectMarkers(frameRight, aruco_dict, parameters=parameters)  # detects marker and gets cornors
            #frame_markers = aruco.drawDetectedMarkers(frameRight.copy(), corners, ids)  # draws the maker in the camera space but details are obscured to the user


            rvecs, tvecs, trash = aruco.estimatePoseSingleMarkers(corners, size_of_marker, mtx, distortion)  # how you get rotation and translation vectors
            imaxis = aruco.drawDetectedMarkers(frameRight.copy(), corners, ids)
            imaxis = cv2.merge((imaxis, imaxis, imaxis))  # Just turn to color for easier display visability

            topLeft, bottomRight = [], []
            ones = np.ones(3)

            if tvecs is not None: #tvecs is not None:
                IDs = np.squeeze(ids)
                for I,i in zip(IDs, range(len(tvecs))):
                    if i == 0:
                        i2 = 0
                        i4 = 1
                        if I == 4:
                            i2 = 1
                            i4 = 0

                        # ID2
                        Rp2 = np.squeeze(rvecs[i2], axis=None)
                        R2, _ = cv2.Rodrigues(Rp2)
                        t2 = np.squeeze(tvecs[i2], axis=None)
                        P_0id2 = my3to4pt(t2)
                        T_cam_id2 = mycomposeRt(R2,t2)
                        T_id2_cam = myinvertT(T_cam_id2)  #invert this

                        # ID4
                        Rp4 = np.squeeze(rvecs[i4], axis=None)
                        R4, _ = cv2.Rodrigues(Rp4)
                        t4 = np.squeeze(tvecs[i4], axis=None)
                        P_0id4 = my3to4pt(t4)
                        T_cam_id4 = mycomposeRt(R4,t4)
                        T_id4_cam = myinvertT(T_cam_id4)

                        # Other points
                        p_m4x = np.array([-length_of_axis * 4, 0., 0.,1.]) # From the point of view of ID2, this is center of ID4
                        p_000 = np.array([0,0,0,1], dtype = float)
                        p_id2_cam = T_id2_cam.dot(p_000) #Camera from id2 frame
                        p_id4_cam = T_id4_cam.dot(p_000) #Camera from id4 frame

                        _P_0id4 = T_cam_id2.dot(p_m4x) #Should be P_0id4 -- point in the center of ID4 in camera frame
                        _p_m4x = T_id2_cam.dot(P_0id4) # Center of ID4_cam should be p_m4x from ID2 coordinate frame
                        #frame id4_from id4
                        T_id2_id4 =  T_id2_cam.dot(T_cam_id4)
                        _p_id2_cam = T_id2_id4.dot(p_id4_cam)

                        print("=============")
                        print("_P_0id4:{} vs P_0id4:{}".format(_P_0id4,P_0id4))
                        print("_p_m4x:{} vs p_m4x:{}".format(_p_m4x,p_m4x))
                        print("_p_id2_cam:{} vs p_id2_cam:{}".format(_p_id2_cam,p_id2_cam))


                    imaxis = cv2.drawFrameAxes(imaxis, mtx, distortion, rvecs[i], tvecs[i], length_of_axis)

                    corner = corners[i]
                    x_mid = (corner[0][1][0] + corner[0][3][0] + corner[0][0][0] + corner[0][2][0]) / 4
                    y_mid = (corner[0][1][1] + corner[0][3][1] + corner[0][0][1] + corner[0][2][1]) / 4
                    topLeft.append(dai.Point2f((x_mid - 15) / mono_width, (y_mid - 15) / mono_height))
                    bottomRight.append(dai.Point2f((x_mid + 15) / mono_width, (y_mid + 15) / mono_height))
                    x_mid = int(x_mid)
                    y_mid = int(y_mid)
                    cv2.circle(imaxis, (x_mid, y_mid), 10, (155, 155, 155), 1)
                    fontType = cv2.FONT_HERSHEY_TRIPLEX
                    cv2.putText(imaxis, f"X: {int(1000*tvecs[i][0][0])} mm",
                                (x_mid + 10, y_mid + 25), fontType, 0.5, color)
                    cv2.putText(imaxis, f"Y: {int(1000*tvecs[i][0][1])} mm",
                                (x_mid + 10, y_mid + 40), fontType, 0.5, color)
                    cv2.putText(imaxis, f"Z: {int(1000*tvecs[i][0][2])} mm",
                                (x_mid + 10, y_mid + 55), fontType, 0.5, color)

            cv2.imshow('Aruco', imaxis)

            if len(topLeft):
                config.roi = dai.Rect(topLeft[0], bottomRight[0])
                cfg = dai.SpatialLocationCalculatorConfig()
                cfg.addROI(config)
                spatialCalcConfigInQueue.send(cfg)
