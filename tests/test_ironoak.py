
from ironoak.person import PersonDetector

videoPath = "/Users/garyding/Public/github_gsoc/depthai-python/examples/models/construction_vest.mp4"
det = PersonDetector.PersonDetector(videoPath)
while True:
    result = det.getResult()             #nonblock read
    print(result)


