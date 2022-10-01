from pathlib import Path

from ironoak.person import person_detector

def test_video():

    videoPath = "/Users/garyding/Public/python-ironoak/src/ironoak/models/construction_vest.mp4"
    nnPath =  str((Path(__file__).parent / Path('../src/ironoak/models/person-detection-retail-0013_openvino_2021.4_7shave.blob')).resolve().absolute())
    det1 = person_detector.person_detector('Video', videoPath, False, nnPath, 0.5)
    while True:
        result1 = det1.getResult()             #nonblock read
        if result1[0]:
            break
    assert result1[0] == True




