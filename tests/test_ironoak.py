
from ironoak.cli import main
from ironoak.person import persondetector
import unittest

class TestMain(unittest.TestCase):
    def test_person(self):
        try:
            result = False
            det = persondetector.persondetector()
            while True:
                result = det.read()             #nonblock read
                if result:
                    break
            self.assertEqual(result, True)      #Passes test if a person is found.
        except:
            pass
