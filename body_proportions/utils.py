
from cameraArray import CamArray
from time import sleep

import json


def loadCamArrayFromJson(json_fname: str):
    json_file = open(json_fname)
    camArrayConfig = json.load(json_file)
    return CamArray((camArrayConfig["left_camera"]["idx"],
                    camArrayConfig["right_camera"]["idx"]))

def loadStereoCameraParameter(json_fname: str):
    json_file = open(json_fname)
    camArrayConfig = json.load(json_file)
    fpx = min(camArrayConfig["left_camera"]["fpx"], camArrayConfig["right_camera"]["fpx"])
    return (camArrayConfig["separation"],fpx,(camArrayConfig["left_camera"]["center"],
                                              camArrayConfig["right_camera"]["center"]))
