
from pathlib import Path
import cv2
import depthai as dai
import numpy as np
import time
import argparse
import threading

class PersonDetector:
    def __init__(self, videoPath):
        self.isDetected = False

        self.x1 = 0
        self.y1 = 0
        self.x2 = 0
        self.y2 = 0

        self.labelMap = ["","person"]

        self.nnPath = str((Path(__file__).parent / Path(
            '../models/person-detection-retail-0013_openvino_2021.4_7shave.blob')).resolve().absolute())
        self.videoPath = videoPath

        # Create pipeline
        self.pipeline = dai.Pipeline()

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
        detectionNetwork.setConfidenceThreshold(0.5)
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
        self.thread = threading.Thread(target=self.detectPerson)
        self.thread.start()

    def getResult(self):
        return (self.isDetected, self.x1, self.y1, self.x2, self.y2)

    def detectPerson(self):
        with dai.Device(self.pipeline) as device:

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
                    print("test")
                    self.x1 = bbox[0]
                    self.y1 = bbox[1]
                    self.x2 = bbox[2]
                    self.y2 = bbox[3]
                    print(self.x1, self.y1, self.x2, self.y2)

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

                trackerFrame = trackFrame.getCvFrame()
                trackletsData = track.tracklets
                for t in trackletsData:
                    roi = t.roi.denormalize(trackerFrame.shape[1], trackerFrame.shape[0])
                    self.x1 = int(roi.topLeft().x)
                    self.y1 = int(roi.topLeft().y)
                    self.x2 = int(roi.bottomRight().x)
                    self.y2 = int(roi.bottomRight().y)
                    # print(self.x1, self.y1, self.x2, self.y2)
                    try:
                        label = self.labelMap[t.label]
                    except:
                        label = t.label
                    if label == "person":
                        self.isDetected = True

                if not trackletsData:
                    self.isDetected = False


