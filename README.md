# indoor-navigation

### Project Description:
This Computer Vision project is designed to assist users who have visual impairments with real-time object detection and navigation through indoor environments. It uses computer vision models to detect objects in the user's path and provides audio feedback about the distance to these objects, helping the user navigate safely. The system uses deep learning models such as Mask R-CNN and SSD MobileNet to detect objects in real-time, along with voice output to provide distance information.

Proposal:
https://docs.google.com/document/d/1qwQG9fdPJrLNisuPtRi_b5iS8Du8wQOIjXV9LccFeG8/edit?usp=share_link

Final Report:
https://docs.google.com/document/d/1l0_b7f9rndC0MSK-LKZ7cPSUEuzKVJ-LSsEnwzPj4s4/edit?usp=sharing

---

### Installation and Execution Instructions

1. **Install TensorFlow Object Detection API:**
   First, clone the TensorFlow Models repository and install the Object Detection API. Follow the installation steps provided in the [TensorFlow Models GitHub repository](https://github.com/tensorflow/models).

2. **Install Google Protobuf and Convert Proto Files:**
   Download the Google Protobuf releases and convert the `.proto` files into Python files using the following command:
   ```bash
   protoc -I=$SRC_DIR --python_out=$DST_DIR $SRC_DIR/addressbook.proto
   ```
   Ensure that the path to the `protoc` binary is added to your system's environment variables.

3. **Run Object Detection with Mask R-CNN (Real-time Object Detection):**
   For enhanced accuracy in object detection, run the code in the `object_detection_webcam.py` script. This script uses the Mask R-CNN model, optimized for better performance, and performs real-time object detection.

4. **Real-time Object Detection with Distance and Voice Output:**
   To integrate distance and voice output, run the `webcam_blind_voice.py` script. This script uses the SSD MobileNet model for real-time object detection and adds distance estimation along with voice feedback. 
   
   If you wish to use a different model, you can modify the code accordingly. A list of available models can be found on the [TensorFlow Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).

   **Note:** If you have a GPU-enabled system, the detection process will be significantly faster and more efficient.

---

### Datasets used:
- COCO: https://www.tensorflow.org/datasets/catalog/coco
- MIT Indoor Scene Recognition: https://web.mit.edu/torralba/www/indoor.html
- KITTI: https://www.kaggle.com/datasets/klemenko/kitti-dataset

---
### Credits & Team Members:
- Himnish Jain
- Vihaan Khare
- Yishu Ma
- Haotian Wang
- [Medium Article](https://medium.com/beingryaan/real-time-object-detection-along-with-distance-and-voice-alerts-for-blinds-a-blind-assistance-1708b97c3ecc) referenced
