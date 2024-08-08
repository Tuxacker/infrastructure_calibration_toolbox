# Infrastructure Calibration Toolbox

A Python toolbox for automated calibration of infrastructure sensors, as described in [our paper](https://arxiv.org/abs/2304.10814).

## Installation

This project depends on [Excalibur](https://github.com/uulm-mrm/excalibur).
A Python 3.10 venv is used. The following commands are used on Ubuntu:

```bash
sudo apt install python3-tk
git clone https://github.com/Tuxacker/infrastructure_calibration_toolbox.git
cd infrastructure_calibration_toolbox
git clone https://github.com/uulm-mrm/excalibur.git
pip install -r excalibur/requirements.txt
pip install -e excalibur/.
pip install --editable .
```

## Running the calibration CLI

Sample data is provided, run the example with:

```
ENABLE_CALIBRATION=1 python3 infrastructure_tools/calibration/autocalibration.py -a data/imu_data.yaml -d data/detection_data_camera_02.yaml -y config/ac_intersection_spu.yaml -s 0 -c 2
```

If the input data is changed, the pickle caches need to be manually removed before running the script again.

The detection data format is as follows:

- Key: Timestamp in nanoseconds as integer
- Value: Bounding box center x and y, width, height, object class

The location data format is as follows:

- Key: Timestamp in nanoseconds as integer
- Value: UTM x, y, and z with offset to avoid large values, yaw (0.0 == UTM East), pitch, roll
