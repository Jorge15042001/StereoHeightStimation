
from .cameraArray import CamArray
from typing import List

import json
from dataclasses import dataclass


@dataclass
class CameraConfig:
    idx: int
    fpx: float()
    center: List[float]


@dataclass
class StereoConfig:
    left_camera: CameraConfig
    right_camera: CameraConfig

    cam_separation: float
    stereo_map_file: str
    depth_to_pixel_size: float
    show_images: bool

    save_images: bool

    save_path: str
    resolution: tuple[int, int]


def startCameraArray(stereo_config: StereoConfig) -> CamArray:
    if stereo_config.save_images:
        return CamArray((stereo_config.left_camera.idx,
                         stereo_config.right_camera.idx),
                        save_frames_to=stereo_config.save_path)

    return CamArray((stereo_config.left_camera.idx,
                     stereo_config.right_camera.idx))


def loadStereoCameraConfig(json_fname: str) -> StereoConfig:
    json_file = open(json_fname)
    stero_config = json.load(json_file)

    sep = stero_config["separation"]
    stereo_map_file = stero_config["stereo_map_file"]
    depth_to_pixel_size = stero_config["depth_to_pixel_size"]
    show_images = stero_config["show_images"]

    save_imgs = stero_config["save_images"]

    save_imgs_path = stero_config["save_path"]
    resx, resy = stero_config["resolution"]
    res = (resx, resy)

    left_camera = CameraConfig(
        stero_config["left_camera"]["idx"],
        stero_config["left_camera"]["fpx"],
        stero_config["left_camera"]["center"]
    )

    right_camera = CameraConfig(
        stero_config["right_camera"]["idx"],
        stero_config["right_camera"]["fpx"],
        stero_config["right_camera"]["center"]
    )

    return StereoConfig(left_camera, right_camera, sep,
                        stereo_map_file, depth_to_pixel_size,
                        show_images, save_imgs,  save_imgs_path, res)
