from pathlib import Path

from ironoak.person import person_detector

def test_camera():
    modelPath = str((Path(__file__).parent / Path(
        '../src/ironoak/models/mobilenet-ssd_openvino_2021.4_5shave.blob')).resolve().absolute())
    det = person_detector.person_detector('Camera', None, False, modelPath, 0.5)
    while True:
        result = det.getResult()             #nonblock read
        if result[0]:
            break
    assert result[0] == True



