# StereoHeightEstimation

This project implements a framwork for estimading people height using streo vision

## Dependencies
```bash
pipenv install
pipenv shell # enter shell with dependencies
```
## Calibration

### Create calibration folder

```bash
mkdir -pv stereo_config/calibration_images
cp  height_estimation/stereo_config.json stereo_config/
# inside stereo_config.json set the camera id of each camera and the separation between cameras
```

### Capture calibration images

```bash
python -m height_estimation.calibration_images stereo_config/stereo_config.json stereo_config/calibration_images/

# Press 'q' to exit
# Press 's' to preview captured image
#   Press 'q' to discard previewed image
#   Press 's' to save previewed image
```





### Perform calibration

```bash
python -m height_estimation.stereo_calibration stereo_config/calibration_images/ stereo_config/stereoMap.xml stereo_config/stereo_config.json


```
Update "stereo_map_file" key in stereo_config/stereo_config.json to either an absolute path or path relative to the root of the repository
