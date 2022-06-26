
from ironoak.person import person_detector

def test_camera():
    det = person_detector.person_detector(None, False)
    while True:
        result = det.getResult()             #nonblock read
        if result[0]:
            break
    assert result[0] == True



