# Head Pose Estimation - OpenCV

In computer vision pose estimation specifically refers to the relative orientation of the object with respect to a camera. Pose estimation often referred to as a Perspective-n-Point problem or PNP problem in computer vision.

![](sample.gif)<br>
sample video [credits](https://www.pexels.com/video/close-up-of-a-woman-showing-different-facial-expressions-3063839/)

## Installation

Using the package manager [pip](https://pip.pypa.io/en/stable/).

```bash
pip install -r requirements.txt
```
```bash
cd models
bash downloader.sh
cd ..
```
## Usage
#### Get Pose From Image
```
python head_pose_from_image.py -h
```
#### Get Pose From Webcame
```
python head_pose_from_webcam.py -h

#### For source 0 and focal length 1
python head_pose_from_webcam.py -f 1 -s 0
```
#### 3D model visualization
```
python Visualize3DModel.py
```
