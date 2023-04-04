
from cameraArray import CamArray
from time import sleep

import json


def loadCamArrayFromJson(json_fname: str):
    json_file = open(json_fname)
    camArrayConfig = json.load(json_file)
    return CamArray((camArrayConfig["left_camera"]["idx"],
                    camArrayConfig["right_camera"]["idx"]))


