
# YOLO-ViT Material Detector

This ROS package integrates YOLO object detection with Vision Transformer (ViT) based material classification. It processes images from a ROS topic, detects objects using YOLO, and classifies the material of the detected objects using a ViT model fine-tuned on the MINC-2500 dataset.

## Requirements

- ROS (tested on Noetic)
- OpenCV (version 4.10.0 or higher)
- NumPy
- cv_bridge
- ultralytics (YOLO)
- transformers (Hugging Face)

## Installation

1. **Clone the Repository**:
   \`\`\`bash
   git clone https://github.com/your-username/your-forked-repo.git
   cd your-forked-repo
   \`\`\`

2. **Install Dependencies**:
   Make sure you have the necessary ROS packages and Python libraries installed.

   \`\`\`bash
   sudo apt-get install ros-noetic-cv-bridge ros-noetic-sensor-msgs
   pip install opencv-python-headless numpy ultralytics transformers
   \`\`\`

## Usage

1. **Build the ROS Package**:
   Make sure your ROS workspace is set up correctly. Place the package in the `src` directory of your workspace and build it.

   \`\`\`bash
   cd ~/catkin_ws
   catkin_make
   source devel/setup.bash
   \`\`\`

2. **Run the Node**:
   Launch the ROS node that runs the YOLO-ViT material detector.

   \`\`\`bash
   rosrun your_package_name yolo_vit_mat_detector.py
   \`\`\`

## Node Details

### Subscribed Topics

- \`/camera/image_raw\` (\`sensor_msgs/Image\`): The raw image topic from the camera.

### Published Topics

- \`vit_inference/result\` (\`vit_inference/MaterialDetected\`): The detected material information.
- \`vit_inference/image\` (\`sensor_msgs/Image\`): The image with annotated detections.

## Code Overview

- **Initialization**: The `YoloVitMaterialDetector` class initializes the ROS node, subscribes to the image topic, and sets up publishers. It also loads the YOLO model for object detection and the Vision Transformer model for material classification.

- **Image Callback**: The `image_callback` method processes each incoming image:
  - Converts the ROS image message to an OpenCV image.
  - Uses YOLO to detect objects in the image.
  - Crops each detected object and uses the ViT model to classify the material.
  - Publishes the detected material and annotated image.

## Example Output

The node prints timing information for various stages of the processing pipeline and the detected material and confidence.

## References

- [YOLO by Ultralytics](https://github.com/ultralytics/yolov5)
- [Vision Transformer by Hugging Face](https://huggingface.co/models?filter=vit)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
