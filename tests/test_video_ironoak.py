
from ironoak.person import person_detector

def test_video():

    videoPath = "/Users/garyding/Public/github_gsoc/depthai-python/examples/models/construction_vest.mp4"
    det1 = person_detector.person_detector(videoPath, None)
    while True:
        result1 = det1.getResult()             #nonblock read
        if result1[0]:
            break
    assert result1[0] == True




