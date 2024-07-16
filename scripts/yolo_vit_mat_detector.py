#!/usr/bin/env python3

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO
from transformers import ViTForImageClassification, ViTImageProcessor
from transformers import AutoImageProcessor, AutoModelForImageClassification

from vit_inference.msg import MaterialDetected

#from ultralytics import RTDETR



# # Use a pipeline as a high-level helper OR Load model directly (BELOW)
# from transformers import pipelinefrom transformers import AutoImageProcessor, AutoModelForImageClassification


# pipe = pipeline("image-classification", model="ioanasong/vit-MINC-2500")


# Load model directly
processor = AutoImageProcessor.from_pretrained("ioanasong/vit-MINC-2500")
model = AutoModelForImageClassification.from_pretrained("ioanasong/vit-MINC-2500") 

# labels_materials = ['brick', 'carpet','ceramic','fabric','foliage', 'food','glass','hair',
#                     'leather','metal','mirror','other','painted','paper','plastic','polishedstone',
#                     'skin','sky','stone','tile','wallpaper','water','wood']


class YoloVitMaterialDetector:
    def __init__(self):
        self.bridge = CvBridge()
        self.labels_materials = ['brick', 'carpet', 'ceramic', 'fabric', 'foliage', 'food', 'glass', 'hair',
                             'leather', 'metal', 'mirror', 'other', 'painted', 'paper', 'plastic', 'polishedstone',
                             'skin', 'sky', 'stone', 'tile', 'wallpaper', 'water', 'wood']
        # self.yolo_model = YOLO('yolov10n.pt')  # "Suitable for extremely resource-constrained environments" for object-detection
        self.yolo_model = YOLO('yolov10s.pt')  # "Balances speed and accuracy" for object-detection
        # self.rtdetr_model = RTDETR("rtdetr-l.pt") # for OD

            # results = model("image.jpg")
            # results[0].show()
        # self.vit_model = ViTForImageClassification.from_pretrained('vit-MINC-2500')
        self.vit_model = model
        # self.vit_processor = ViTImageProcessor.from_pretrained('vit-MINC-2500')
        self.vit_processor = processor
        self.image_sub = rospy.Subscriber('/camera/image_raw', Image, self.image_callback) # TODO check which subscriber is camera for '/camera/image_raw'
        self.result_pub = rospy.Publisher('vit_inference/result', MaterialDetected, queue_size=10)  #TODO subscribe to this
        self.result_image = rospy.Publisher('vit/inference/result', Image, queue_size=10)
        print("Initialized Detector")

    def image_callback(self, msg):

        print("STARTED Image callback")
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "8UC3")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        # YOLO detection
        # print("YOLO detection starting")

        results = self.yolo_model(cv_image) 
        
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                crop = cv_image[y1:y2, x1:x2]
                
                # Vision Transformer inference
                # print("ViT inference starting")
                inputs = self.vit_processor(images=crop, return_tensors="pt")
                outputs = self.vit_model(**inputs)
                predicted_class = outputs.logits.argmax(-1).item()
                confidence = outputs.logits.softmax(-1).max().item()

                # Saving values relevant about material detected and class
                # print("Saving material detected results of a box")
                det_msg = MaterialDetected()
                det_msg.header = msg.header
                det_msg.object_class = result.names[int(result.boxes.cls[0])]
                det_msg.confidence = confidence
                det_msg.x = x1
                det_msg.y = y1
                det_msg.width = x2-x1
                det_msg.height = y2-y1
                det_msg.material = self.labels_materials[predicted_class]

                self.result_pub.publish(det_msg)

                # Draw bounding box and label
                # print("Drawing bounding box and label")
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(cv_image, f"Class: {self.labels_materials[predicted_class]}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        if len(results) > 0 and len(results[0].boxes) > 0:
            cv2.imshow("Material Detection", cv_image)
            cv2.waitKey(1)
            # Publish the image with detections
            img_msg = self.bridge.cv2_to_imgmsg(cv_image, "8UC3")
            self.result_image.publish(img_msg)

            # print("Material detected: "+ str(predicted_class) + " "+ labels_materials[predicted_class] + " "+ det_msg.object_class)
            print(f"Material detected: {predicted_class} {self.labels_materials[predicted_class]} {det_msg.object_class}")
            print("Confidence of material detection: "+ str(confidence))

        else:
            print("No objects detected")
            # predicted_class = -1
            # det_msg.material = "unknown"

                

if __name__ == '__main__':
    rospy.init_node('yolo_vit_mat_detector', anonymous=True)
    detector = YoloVitMaterialDetector()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()