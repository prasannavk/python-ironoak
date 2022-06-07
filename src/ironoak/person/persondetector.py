
from pathlib import Path
import cv2
import depthai as dai
import numpy as np
import time
import argparse
import threading

class persondetector:
    def __init__(self):
        self.isdetected = False

        self.labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

        nnPathDefault = str((Path(__file__).parent / Path('../models/mobilenet-ssd_openvino_2021.4_6shave.blob')).resolve().absolute())
        parser = argparse.ArgumentParser()
        parser.add_argument('nnPath', nargs='?', help="Path to mobilenet detection network blob", default=nnPathDefault)
        parser.add_argument('-ff', '--full_frame', action="store_true", help="Perform tracking on full RGB frame", default=False)

        args = parser.parse_args()

        fullFrameTracking = args.full_frame

        # Create pipeline
        self.pipeline = dai.Pipeline()

        # Define sources and outputs
        camRgb = self.pipeline.create(dai.node.ColorCamera)
        detectionNetwork = self.pipeline.create(dai.node.MobileNetDetectionNetwork)
        objectTracker = self.pipeline.create(dai.node.ObjectTracker)

        xlinkOut = self.pipeline.create(dai.node.XLinkOut)
        trackerOut = self.pipeline.create(dai.node.XLinkOut)

        xlinkOut.setStreamName("preview")
        trackerOut.setStreamName("tracklets")

        # Properties
        camRgb.setPreviewSize(300, 300)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camRgb.setInterleaved(False)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        camRgb.setFps(40)

        # testing MobileNet DetectionNetwork
        detectionNetwork.setBlobPath(args.nnPath)
        detectionNetwork.setConfidenceThreshold(0.5)
        detectionNetwork.input.setBlocking(False)

        objectTracker.setDetectionLabelsToTrack([15])  # track only person
        # possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS, SHORT_TERM_IMAGELESS, SHORT_TERM_KCF
        objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
        # take the smallest ID when new object is tracked, possible options: SMALLEST_ID, UNIQUE_ID
        objectTracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)

        # Linking
        camRgb.preview.link(detectionNetwork.input)
        objectTracker.passthroughTrackerFrame.link(xlinkOut.input)

        if fullFrameTracking:
            camRgb.video.link(objectTracker.inputTrackerFrame)
        else:
            detectionNetwork.passthrough.link(objectTracker.inputTrackerFrame)

        detectionNetwork.passthrough.link(objectTracker.inputDetectionFrame)
        detectionNetwork.out.link(objectTracker.inputDetections)
        objectTracker.out.link(trackerOut.input)
        self.thread = threading.Thread(target=self.detectperson, args=())
        self.thread.start()

    def read(self):
        isdetected = self.isdetected
        return isdetected

    def detectperson(self):
        with dai.Device(self.pipeline) as device:

            preview = device.getOutputQueue("preview", 4, False)
            tracklets = device.getOutputQueue("tracklets", 4, False)

            startTime = time.monotonic()
            counter = 0

            while (True):
                imgFrame = preview.get()
                track = tracklets.get()
                counter += 1
                current_time = time.monotonic()
                if (current_time - startTime) > 1:
                    counter = 0
                    startTime = current_time

                frame = imgFrame.getCvFrame()
                trackletsData = track.tracklets
                for t in trackletsData:
                    roi = t.roi.denormalize(frame.shape[1], frame.shape[0])
                    x1 = int(roi.topLeft().x)
                    y1 = int(roi.topLeft().y)
                    x2 = int(roi.bottomRight().x)
                    y2 = int(roi.bottomRight().y)

                    try:
                        label = self.labelMap[t.label]
                    except:
                        label = t.label

                    if label == "person":
                        self.isdetected = True

                if not trackletsData:
                    self.isdetected = False
