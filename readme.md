# MEAL (Multitask Efficient trAnsformer network for Laryngoscopy)

This repository contains the implementation of the MEAL model, a Multitask Efficient trAnsformer network for Laryngoscopy. The MEAL model is designed for the classification of vocal fold images and the detection of glottic landmarks and lesions. The paper will be available soon.

## Installation
1. Clone this repository and create a new conda environment:
```
git clone https://github.com/LouisDo2108/MEAL.git
conda create -n meal python=3.8.15
cd meal
conda activate meal
```

2. Install the required packages:
```
pip install -r requirements.txt
```

3. Download the pretrained weights and data from here: [google_drive](https://drive.google.com/drive/folders/1B6hFoUSijcOSNkZafNPULgnJ-jkQOqX0?usp=share_link)

4. Unzip the downloaded files and place them under the repository folder.

5. You can start doing things like training and validating the backbone or the whole model. Please see some example scripts in run.sh

## Acknowledgements
This implementation of the MEAL model draws inspiration from the YOLOv5 and SCET architectures. We would like to acknowledge the contributions of these projects.

## References
- [YOLOv5](https://github.com/ultralytics/yolov5)
- [SCET](https://github.com/AlexZou14/SCET)