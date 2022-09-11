
from pathlib import Path
import streamlit as st

import cv2
import depthai as dai
import numpy as np
import time
from networktables import NetworkTables
import json
import argparse
import threading


class person_detector:
    def configPs(self):
        self.isDetected = False
        self.fullFrameTracking = False
        self.x1 = 0
        self.y1 = 0
        self.x2 = 0
        self.y2 = 0

        self.depth = 0

        if self.inputSrc == "Camera":
            self.labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
        else:
            self.labelMap = ["", "person"]

        # Create pipeline
        self.pipeline = dai.Pipeline()

        NetworkTables.initialize(server='192.168.1.252')
        self.sd = NetworkTables.getTable("SmartDashboard")

        if self.inputSrc == "Camera":
            camRgb = self.pipeline.create(dai.node.ColorCamera)
            spatialDetectionNetwork = self.pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
            monoLeft = self.pipeline.create(dai.node.MonoCamera)
            monoRight = self.pipeline.create(dai.node.MonoCamera)
            stereo = self.pipeline.create(dai.node.StereoDepth)
            objectTracker = self.pipeline.create(dai.node.ObjectTracker)

            # xoutRgb = self.pipeline.create(dai.node.XLinkOut)
            xoutRgb = self.pipeline.createXLinkOut()
            trackerOut = self.pipeline.create(dai.node.XLinkOut)

            xoutRgb.setStreamName("preview")
            trackerOut.setStreamName("tracklets")

            # Properties
            camRgb.setPreviewSize(300, 300)
            camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
            camRgb.setInterleaved(False)
            camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

            monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
            monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

            # setting node configs
            stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
            # Align depth map to the perspective of RGB camera, on which inference is done
            stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
            stereo.setOutputSize(monoLeft.getResolutionWidth(), monoLeft.getResolutionHeight())

            spatialDetectionNetwork.setBlobPath(self.nnPath)
            spatialDetectionNetwork.setConfidenceThreshold(self.confThres)
            spatialDetectionNetwork.input.setBlocking(False)
            spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
            spatialDetectionNetwork.setDepthLowerThreshold(100)
            spatialDetectionNetwork.setDepthUpperThreshold(5000)

            objectTracker.setDetectionLabelsToTrack([15])  # track only person
            # possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS, SHORT_TERM_IMAGELESS, SHORT_TERM_KCF
            objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
            # take the smallest ID when new object is tracked, possible options: SMALLEST_ID, UNIQUE_ID
            objectTracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)

            # Linking
            monoLeft.out.link(stereo.left)
            monoRight.out.link(stereo.right)

            camRgb.preview.link(spatialDetectionNetwork.input)
            objectTracker.passthroughTrackerFrame.link(xoutRgb.input)
            objectTracker.out.link(trackerOut.input)

            if self.fullFrameTracking:
                camRgb.setPreviewKeepAspectRatio(False)
                camRgb.video.link(objectTracker.inputTrackerFrame)
                objectTracker.inputTrackerFrame.setBlocking(False)
                # do not block the pipeline if it's too slow on full frame
                objectTracker.inputTrackerFrame.setQueueSize(2)
            else:
                spatialDetectionNetwork.passthrough.link(objectTracker.inputTrackerFrame)

            spatialDetectionNetwork.passthrough.link(objectTracker.inputDetectionFrame)
            spatialDetectionNetwork.out.link(objectTracker.inputDetections)
            stereo.depth.link(spatialDetectionNetwork.inputDepth)

        else:
            # Define sources and outputs
            manip = self.pipeline.create(dai.node.ImageManip)
            objectTracker = self.pipeline.create(dai.node.ObjectTracker)
            detectionNetwork = self.pipeline.create(dai.node.MobileNetDetectionNetwork)

            manipOut = self.pipeline.create(dai.node.XLinkOut)
            xinFrame = self.pipeline.create(dai.node.XLinkIn)
            trackerOut = self.pipeline.create(dai.node.XLinkOut)
            xlinkOut = self.pipeline.create(dai.node.XLinkOut)
            nnOut = self.pipeline.create(dai.node.XLinkOut)

            manipOut.setStreamName("manip")
            xinFrame.setStreamName("inFrame")
            xlinkOut.setStreamName("trackerFrame")
            trackerOut.setStreamName("tracklets")
            nnOut.setStreamName("nn")

            # Properties
            xinFrame.setMaxDataSize(1920 * 1080 * 3)
            manip.initialConfig.setResizeThumbnail(544, 320)
            # manip.initialConfig.setResize(384, 384)
            # manip.initialConfig.setKeepAspectRatio(False) #squash the image to not lose FOV
            # The NN model expects BGR input. By default ImageManip output type would be same as input (gray in this case)
            manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
            manip.inputImage.setBlocking(True)

            # setting node configs
            detectionNetwork.setBlobPath(self.nnPath)
            detectionNetwork.setConfidenceThreshold(self.confThres)
            detectionNetwork.input.setBlocking(True)

            objectTracker.inputTrackerFrame.setBlocking(True)
            objectTracker.inputDetectionFrame.setBlocking(True)
            objectTracker.inputDetections.setBlocking(True)
            objectTracker.setDetectionLabelsToTrack([1])  # track only person
            # possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS, SHORT_TERM_IMAGELESS, SHORT_TERM_KCF
            objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
            # take the smallest ID when new object is tracked, possible options: SMALLEST_ID, UNIQUE_ID
            objectTracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)

            # Linking
            manip.out.link(manipOut.input)
            manip.out.link(detectionNetwork.input)
            xinFrame.out.link(manip.inputImage)
            xinFrame.out.link(objectTracker.inputTrackerFrame)
            detectionNetwork.out.link(nnOut.input)
            detectionNetwork.out.link(objectTracker.inputDetections)
            detectionNetwork.passthrough.link(objectTracker.inputDetectionFrame)
            objectTracker.out.link(trackerOut.input)
            objectTracker.passthroughTrackerFrame.link(xlinkOut.input)

    def getResult(self):
        if self.videoPath is None:
            return (self.isDetected, self.x1, self.y1, self.x2, self.y2, self.depth)
        else:
            return (self.isDetected, self.x1, self.y1, self.x2, self.y2)

    def detectPerson(self):

        with dai.Device(self.pipeline) as device:

            frameST = st.empty()
            device.startPipeline()

            if self.inputSrc == "Video":
                qIn = device.getInputQueue(name="inFrame")
                trackerFrameQ = device.getOutputQueue(name="trackerFrame", maxSize=4)
                tracklets = device.getOutputQueue(name="tracklets", maxSize=4)
                qManip = device.getOutputQueue(name="manip", maxSize=4)
                qDet = device.getOutputQueue(name="nn", maxSize=4)

                startTime = time.monotonic()
                counter = 0
                fps = 0
                detections = []
                frame = None

                def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
                    return cv2.resize(arr, shape).transpose(2, 0, 1).flatten()

                # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
                def frameNorm(frame, bbox):
                    normVals = np.full(len(bbox), frame.shape[0])
                    normVals[::2] = frame.shape[1]
                    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

                def displayFrame(name, frame):
                    for detection in detections:
                        bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                        self.x1 = bbox[0]
                        self.y1 = bbox[1]
                        self.x2 = bbox[2]
                        self.y2 = bbox[3]

                cap = cv2.VideoCapture(self.videoPath)
                baseTs = time.monotonic()
                simulatedFps = 30
                inputFrameShape = (1280, 720)

                while cap.isOpened():
                    read_correctly, frame = cap.read()
                    if not read_correctly:
                        break

                    img = dai.ImgFrame()
                    img.setType(dai.ImgFrame.Type.BGR888p)
                    img.setData(to_planar(frame, inputFrameShape))
                    img.setTimestamp(baseTs)
                    baseTs += 1 / simulatedFps

                    img.setWidth(inputFrameShape[0])
                    img.setHeight(inputFrameShape[1])
                    qIn.send(img)

                    trackFrame = trackerFrameQ.tryGet()
                    if trackFrame is None:
                        continue

                    track = tracklets.get()
                    manip = qManip.get()
                    inDet = qDet.get()

                    detections = inDet.detections
                    manipFrame = manip.getCvFrame()


                    color = (255, 0, 0)
                    trackerFrame = trackFrame.getCvFrame()
                    trackletsData = track.tracklets

                    #frameST = st.empty()
                    for t in trackletsData:
                        roi = t.roi.denormalize(trackerFrame.shape[1], trackerFrame.shape[0])
                        self.x1 = int(roi.topLeft().x)
                        self.y1 = int(roi.topLeft().y)
                        self.x2 = int(roi.bottomRight().x)
                        self.y2 = int(roi.bottomRight().y)

                        try:
                            label = self.labelMap[t.label]
                        except:
                            label = t.label
                        #if label == "person":
                        #    return True, self.x1, self.x2, self.y1, self.y2

                        cv2.rectangle(trackerFrame, (self.x1, self.y1), (self.x2, self.y2), color, cv2.FONT_HERSHEY_SIMPLEX)

                    # with st.empty():
                    frameST.image(trackerFrame, channels='BGR')
                    #     time.sleep(0.05)

                    if not trackletsData:
                        self.isDetected = False
                        self.x1 = 0
                        self.y1 = 0
                        self.x2 = 0
                        self.y2 = 0

            else:
                preview = device.getOutputQueue("preview", 4, False)
                tracklets = device.getOutputQueue("tracklets", 4, False)

                startTime = time.monotonic()
                counter = 0
                fps = 0
                color = (255, 255, 255)

                # frameST = st.empty()

                while (True):
                    imgFrame = preview.get()
                    track = tracklets.get()

                    counter += 1
                    current_time = time.monotonic()
                    if (current_time - startTime) > 1:
                        fps = counter / (current_time - startTime)
                        counter = 0
                        startTime = current_time

                    frame = imgFrame.getCvFrame()
                    trackletsData = track.tracklets
                    for t in trackletsData:
                        roi = t.roi.denormalize(frame.shape[1], frame.shape[0])
                        self.x1 = int(roi.topLeft().x)
                        self.y1 = int(roi.topLeft().y)
                        self.x2 = int(roi.bottomRight().x)
                        self.y2 = int(roi.bottomRight().y)
                        self.depth = int(t.spatialCoordinates.z)

                        try:
                            label = self.labelMap[t.label]
                        except:
                            label = t.label

                        cv2.rectangle(frame, (self.x1, self.y1), (self.x2, self.y2), color, cv2.FONT_HERSHEY_SIMPLEX)
                        cv2.putText(frame, f"Depth: {int(t.spatialCoordinates.z)} mm", (self.x1 + 10, self.y1 + 95),
                                    cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

                        data = {}
                        data['x1'] = self.x1
                        data['y1'] = self.y1
                        data['x2'] = self.x2
                        data['y2'] = self.y2
                        data['depth'] = self.depth
                        json_data = json.dumps(data)
                        self.sd.putString("Person Detector", json_data)

                        #with frameST.empty():
                        frameST.image(frame, channels='BGR')

                        #if label == "person":
                        #    return True, self.x1, self.x2, self.y1, self.y2, self.depth
                        #else:
                        #    return False, 0, 0, 0, 0, 0
                   # return False, 0,0,0,0,0


if __name__ == '__main__':
    p1 = person_detector()
    # Header
    st.sidebar.header("Iron-oak Person Detector")

    # Subheader
    st.sidebar.subheader("Version 0.0.4 (2022-7-26)")

    p1.videoPath = None
    p1.nnPath = None
    p1.isRunning = False

    # Selection box for input source
    p1.inputSrc = st.sidebar.selectbox("Choose an input source: ", ['Camera', 'Video'])

    if p1.inputSrc == 'Video':
        vidSelect = st.sidebar.selectbox("Choose a video: ", ['Sample', 'Other'])
        if vidSelect == 'Sample':
            p1.videoPath = str((Path(__file__).parent / Path('../models/construction_vest.mp4')).resolve().absolute())
        elif vidSelect == 'Other':
            uploaded_file = st.file_uploader("Choose a file")
            if uploaded_file is not None:
                p1.videoPath = str((Path(__file__).parent / Path('../models/' + uploaded_file.name)).resolve().absolute())
        else:
            p1.videoPath = None

    # Selection box for model file
    nnSelect = st.sidebar.selectbox("Choose a model: ", ['Camera Default', 'Video Default', 'Other'])

    if nnSelect == 'Camera Default':
        p1.nnPath = str(
            (Path(__file__).parent / Path('../models/mobilenet-ssd_openvino_2021.4_5shave.blob')).resolve().absolute())
    elif nnSelect == 'Video Default':
        p1.nnPath = str((Path(__file__).parent / Path(
            '../models/person-detection-retail-0013_openvino_2021.4_7shave.blob')).resolve().absolute())
    else:
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            p1.nnPath = str((Path(__file__).parent / Path('../models/' + uploaded_file.name)).resolve().absolute())

    # Slider for confidence threshold
    p1.confThres = st.sidebar.slider("Select the Confidence Threshold:", 0.001, 0.5, 0.5, 0.001)

    if (st.sidebar.button('Start')):
        p1.configPs()
        p1.isRunning = True

    with st.empty():
        while p1.isRunning:
            st.write(p1.detectPerson())





