import sys
import time
from pathlib import Path
import argparse

import cv2
import depthai as dai
import numpy as np
from cv2 import aruco  # dont know why this is red underlined doesnet seem to have a problem


def ProjectPointRT(R,t,p):
    P = R@p + t
    return P

def ProjectPointRTvecs(rvec,tvec,p):
    R = cv2.Rodrigues(rvec)[0]
    t = np.transpose(tvec)

# tvec_T = np.transpose(tvec)
# R_m = cv2.Rodrigues(rvec_m)[0]
# p = np.array([[2.],[0.],[0.]])

def TransformBetweenMarkers(tvec_m, tvec_n, rvec_m, rvec_n):
    tvec_m = np.transpose(tvec_m)  # tvec of 'm' marker
    tvec_n = np.transpose(tvec_n)  # tvec of 'n' marker
    dtvec = tvec_m - tvec_n  # vector from 'm' to 'n' marker in the camera's coordinate system

    # get the markers' rotation matrices respectively
    R_m = cv2.Rodrigues(rvec_m)[0]
    R_n = cv2.Rodrigues(rvec_n)[0]

    tvec_mm = (-R_m.T).dot(tvec_m)  # np.matmul(-R_m.T, tvec_m) #  camera pose in 'm' marker's coordinate system # one to left
    tvec_nn = (-R_n.T).dot(tvec_n)  # np.matmul(-R_n.T, tvec_n) # camera pose in 'n' marker's coordinate system

    # translational difference between markers in 'm' marker's system,
    # basically the origin of 'n'
    dtvec_m = (-R_m.T).dot(dtvec)  # np.matmul(-R_m.T, dtvec) #  this is the inverse of the rotationb matrix becauethe transpose is the inverse

    # this gets me the same as tvec_mm,
    # but this only works, if 'm' marker is seen
    # tvec_nm = dtvec_m + np.matmul(-R_m.T, tvec_n)

    # something with the rvec difference must give the transformation(???)
    # drvec = rvec_m - rvec_n
    # drvec_m = np.transpose((R_m.T).dot(np.transpose(drvec)))  # np.transpose(np.matmul(R_m.T, np.transpose(drvec)))  # transformed to 'm' marker
    dR_m = R_m.dot(R_n.T)  # cv2.Rodrigues(drvec_m)[0]

    # I want to transform tvec_nn with a single matrix,
    # so it would be interpreted in 'm' marker's system
    tvec_nm = dtvec_m + (dR_m.T).dot(tvec_nn)  # dtvec_m + np.matmul(dR_m.T, tvec_nn) #

    # new
    # cv2.circle(imaxis, (tvec_nm[0], tvec_nm[1]), 5, (0, 0, 255), -1)

    # new:
    m = False
    if (tvec_nm == tvec_mm).any():
        m = True
    #print("tvec_mm:\n{}\ntvec_nm:\n{}\n{}".format(tvec_mm, tvec_nm, m))
    return


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






            # tvecT = np.transpose(tvecs[0])
            # R = cv2.Rodrigues(rvecs[0])[0]
            # p = np.array([[2.], [0.], [0.]])
            # P = ProjectPointRT(R, t, p)
            # print("P is {}".format(P))


            # print("tvecs.shape = {} rvecs.shape(squeeze) = {}\n{}".format(np.shape(tvecs), /n
            # np.shape(np.squeeze(tvecs)), tvecs))
            # new:
            # x = np.matrix([[0, 0, 0], [length_of_axis, 0, 0], [0, length_of_axis, 0], [0, 0, length_of_axis]])
            # x_prime, J = cv2.projectPoints(x, rvecs, tvecs, mtx, dist)

                # objective: tvec_mm == tvec_nm

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
                        #TransformBetweenMarkers(tvecs[i], tvecs[i+1], rvecs[i], rvecs[i+1])
                        # CONVENTION
                        # non-caps == Camera coordinate system, CAPS == world coordinates
                        # ID 2 is aruco ID 2 closer to center of right camera
                        # ID 4 is is acruco to the left of acruco ID 2
                        # V  vector out to the world point (ID 2 or 4)
                        # v result point projected back to the camera
                        # p point I made in camera coordinates
                        # ID2
                        Rv2 = np.squeeze(rvecs[i2], axis=None)
                        R2, _ = cv2.Rodrigues(Rv2)
                        t2 = np.squeeze(tvecs[i2], axis=None)
                        V2 = t2

                        # ID4
                        Rv4 = np.squeeze(rvecs[i4], axis=None)
                        R4, _ = cv2.Rodrigues(Rv4)
                        t4 = np.squeeze(tvecs[i4], axis=None)
                        V4 = t4

                        p4 = np.array([-length_of_axis * 4, 0., 0.])
                        P4 = R2.dot(p4) + t2  # p4 is point I made sitting in center of ID 4 on the camera, P4 is it's projection
                        P4_V2 = P4 - V4
                        print("P4:{}; V4:{}, P4-V2:{}".format(P4, V4, P4_V2))

                        v4 = R2.T.dot(V4 - t2)  #What is camera point V_id4 in id2 coordinate system? It is (-4*Xside, 0,0)
                        p4_v4 = p4 - v4
                        print(" T: v4:{}, p4:{}, p4-v4:{}".format(v4, p4, p4_v4))

                        # V42 = V4 - V2  # vector from acuco board 2 center to acruco board 4 center
                        # PV42_2 = R2.dot(V42) + t2
                        # PV42_4 = R4.dot(PV42_2) #This is zero?
                        # R42 = R4.dot(R2.T) # Projection of id2 to id4
                        # p42_4 = R42.dot(V42)
                        # print("p42_4: {}".format(p42_4))
                        DV = np.abs(P4_V2).dot(ones)
                        dv = np.abs(p4_v4).dot(ones)
                        foo += 1
                        if not foo%10:
                            foo += 1
                        if dv < 0.018 and DV < 0.018:
                            print("GOOD! dv:{}, DV:{}\n_________".format(dv,DV))
                        else:
                            print("***BAD! dv:{}, DV:{}\nxxxxxxxx".format(dv,DV))
                        V1 = np.squeeze(tvecs[0])




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

                        # cv2.circle(imaxis, (165, 171), 5, (0, 0, 255), -1)
                #         rvec = np.squeeze(rvecs[0], axis=None)
                #                 #         tvec = np.squeeze(tvecs[0], axis=None)
                #         tvec = np.expand_dims(tvec, axis=1)
                #         rvec_matrix = cv2.Rodrigues(rvec)[0]
                #
                #         proj_matrix = np.hstack((rvec_matrix, tvec))
                #
                #         # proj_matrix_inv = np.hstack((np.linalg.inv(rvec_matrix),-tvec))
                #         # world_point = np.array([depthData.spatialCoordinates.x,depthData.spatialCoordinates.y,depthData.spatialCoordinates.z,1000.0])/1000.0
                #         # #world_pointT = np.transpose(world_point)
                #         # proj_wpinv = np.matmul(proj_matrix_inv,world_point)
                #         # proj_wp = np.matmul(proj_matrix,world_point)
                #         # print("PT from:\n{}\nPT to inv:\n{}\nPt to orig:\n{}\n______".format(world_point,proj_wpinv,proj_wp))
                #
                #         # print("proj matrix\n",proj_matrix)
                #         euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
                #         cv2.putText(imaxis, 'X: ' + str(int(euler_angles[0])), (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255))
                #         cv2.putText(imaxis, 'Y: ' + str(int(euler_angles[1])), (115, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
                #         cv2.putText(imaxis, 'Z: ' + str(int(euler_angles[2])), (200, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0))
            cv2.imshow('Aruco', imaxis)

            if len(topLeft):
                config.roi = dai.Rect(topLeft[0], bottomRight[0])
                cfg = dai.SpatialLocationCalculatorConfig()
                cfg.addROI(config)
                spatialCalcConfigInQueue.send(cfg)
