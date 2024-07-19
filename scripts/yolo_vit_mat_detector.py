#!/usr/bin/env python3

import rospy
import cv2
import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO
from transformers import ViTForImageClassification, ViTImageProcessor
from transformers import AutoImageProcessor, AutoModelForImageClassification
from vit_inference.msg import MaterialDetected
#from ultralytics import RTDETR


import torch
from torchinfo import summary
from ptflops import get_model_complexity_info

# Loading vision transformer model fine-tuned on the MINC2500 subset of the MINC dataset
processor = AutoImageProcessor.from_pretrained("ioanasong/vit-MINC-2500")
model = AutoModelForImageClassification.from_pretrained("ioanasong/vit-MINC-2500") 

class YoloVitMaterialDetector:
    def __init__(self):
        self.bridge = CvBridge()
        self.labels_materials = ['brick', 'carpet', 'ceramic', 'fabric', 'foliage', 'food', 'glass', 'hair',
                             'leather', 'metal', 'mirror', 'other', 'painted', 'paper', 'plastic', 'polishedstone',
                             'skin', 'sky', 'stone', 'tile', 'wallpaper', 'water', 'wood']
        # self.yolo_model = YOLO('yolov10n.pt')  # "Suitable for extremely resource-constrained environments" for object-detection
        self.yolo_model = YOLO('yolov10s.pt')  # "Balances speed and accuracy" for object-detection
        # self.rtdetr_model = RTDETR("rtdetr-l.pt") # for OD
        self.vit_model = model
        self.vit_processor = processor
        self.image_sub = rospy.Subscriber('/camera/image_raw', Image, self.image_callback)
        self.result_pub = rospy.Publisher('vit_inference/result', MaterialDetected, queue_size=1) 
        self.result_image = rospy.Publisher('vit_inference/image', Image, queue_size=1)
        self.timing_list = []
        self.max_timing_records = 100  # Store last 100 timing records
        print("Initialized Detector")
        # self.calculate_combined_complexity()

    # def calculate_combined_complexity(self):
    #     try:
    #         # YOLO complexity and parameters
    #         yolo_macs, yolo_params = get_model_complexity_info(
    #             self.yolo_model.model, 
    #             (3, 640, 640),  # Adjust if your YOLO input size is different
    #             as_strings=False,  # We want numbers, not strings
    #             print_per_layer_stat=False,
    #             verbose=False
    #         )

    #         # ViT complexity and parameters
    #         vit_macs, vit_params = get_model_complexity_info(
    #             self.vit_model, 
    #             (3, 224, 224),  # Adjust if your ViT input size is different
    #             as_strings=False,
    #             print_per_layer_stat=False,
    #             verbose=False
    #         )

    #         # Calculate total
    #         total_macs = yolo_macs + vit_macs
    #         total_params = yolo_params + vit_params

    #         # Convert to more readable format
    #         total_gflops = total_macs * 2 / 1e9  # Multiply by 2 to convert MACs to FLOPs
    #         total_params_millions = total_params / 1e6

    #         rospy.loginfo(f"Combined Model Complexity:")
    #         rospy.loginfo(f"Total Computational Complexity: {total_gflops:.2f} GFLOPs")
    #         rospy.loginfo(f"Total Parameters: {total_params_millions:.2f} Million")

    #         # Individual model breakdowns
    #         rospy.loginfo(f"YOLO Complexity: {yolo_macs * 2 / 1e9:.2f} GFLOPs")
    #         rospy.loginfo(f"YOLO Parameters: {yolo_params / 1e6:.2f} Million")
    #         rospy.loginfo(f"ViT Complexity: {vit_macs * 2 / 1e9:.2f} GFLOPs")
    #         rospy.loginfo(f"ViT Parameters: {vit_params / 1e6:.2f} Million")

    #     except Exception as e:
    #         rospy.logerr(f"Error calculating combined complexity: {e}")


    def image_callback(self, msg):

        start_total = time.perf_counter()

        print("Started Image callback")
        try:
            # cv_image = self.bridge.imgmsg_to_cv2(msg, "8UC3")
            cv_image = self.bridge.imgmsg_to_cv2(msg, "passthrough")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        # YOLO detection
        start_yolo = time.perf_counter()
        results = self.yolo_model(cv_image) 
        end_yolo = time.perf_counter()
        
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            for box in boxes:

                start_preprocess = time.perf_counter()  # timing ViT preprocess of each bounding box from YOLO
                x1, y1, x2, y2 = map(int, box[:4])
                crop = cv_image[y1:y2, x1:x2]
                end_preprocess = time.perf_counter()

                # Vision Transformer inference
                start_inference = time.perf_counter()   # timing ViT inference duration on each BB
                inputs = self.vit_processor(images=crop, return_tensors="pt")
                outputs = self.vit_model(**inputs)
                predicted_class = outputs.logits.argmax(-1).item()
                confidence = outputs.logits.softmax(-1).max().item()
                end_inference = time.perf_counter()
                
                # Saving values relevant about material detected and class
                message_time = time.perf_counter()
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
                message_end_time = time.perf_counter()

                # Draw bounding box and label
                start_postprocess = time.perf_counter()
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)


                text = f"Class: {self.labels_materials[predicted_class]}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.9
                color = (0, 255, 0)  # Green color
                thickness = 2
                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

                text_x = x1 + 5  # 5 pixels from the left edge of the box
                text_y = y1 + text_height + 5  # 5 pixels from the top edge of the box, plus the height of the text

                if text_x + text_width > x2:
                    text_x = x2 - text_width - 5
                if text_y > y2:
                    text_y = y2 - 5

                cv2.putText(cv_image, text, (text_x, text_y), font, font_scale, color, thickness)

                print(f"Material detected: {predicted_class} {self.labels_materials[predicted_class]} {det_msg.object_class}")
                print("Confidence of material detection: "+ str(confidence))
                
                # cv2.putText(cv_image, f"Class: {self.labels_materials[predicted_class]}", (x1, y1-10),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                end_postprocess = time.perf_counter()
                
                # Calculate speeds in milliseconds
                preprocess_duration = (end_preprocess - start_preprocess) * 1000
                vit_duration = (end_inference - start_inference) * 1000 
                vit_message_duration = (message_end_time - message_time) * 1000
                vit_object_postprocess_duration = (end_postprocess - start_postprocess) *1000

        # Show processed image with detection when objects have been detected
        if len(results) > 0 and len(results[0].boxes) > 0:
            cv2.imshow("Material Detection", cv_image)
            cv2.imwrite('/home/kai/catkin_ws/src/vit_inference/results/images/image' + str(len(self.timing_list)) + '.png', cv_image)
            cv2.waitKey(1)

            # Publish the image with detections
            # img_msg = self.bridge.cv2_to_imgmsg(cv_image, "8UC3")
            # img_msg = self.bridge.cv2_to_imgmsg(cv_image, "passthrough")
            img_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            self.result_image.publish(img_msg)

            end_total = time.perf_counter()
            yolo_duration = (end_yolo - start_yolo) * 1000
            total_duration = (end_total - start_total) * 1000

            timing_info = {
                'preprocess': preprocess_duration,
                'yolo': yolo_duration,
                'vit': vit_duration,
                'message': vit_message_duration,
                'postprocess': vit_object_postprocess_duration,
                'total': total_duration
            }
            self.timing_list.append(timing_info)
            if len(self.timing_list) > self.max_timing_records:
                self.timing_list.pop(0)

            avg_times = {k: sum(t[k] for t in self.timing_list) / len(self.timing_list) 
                        for k in self.timing_list[0].keys()}
            
            print(f"Speed: {preprocess_duration:.1f}ms preprocess, {yolo_duration:.1f}ms YOLO, "
                f"{vit_duration:.1f}ms ViT, {vit_message_duration:.1f}ms publish, "
                f"{vit_object_postprocess_duration:.1f}ms postprocess per image at shape {cv_image.shape}",
                f"{total_duration:.1f}ms total YOLO and ViT image frame processing")
            
            if len(self.timing_list) % 10 == 0:
                print(f"Average Speed: {avg_times['preprocess']:.1f}ms preprocess, "
                    f"{avg_times['yolo']:.1f}ms YOLO, {avg_times['vit']:.1f}ms ViT, "
                    f"{avg_times['message']:.1f}ms publishing, {avg_times['postprocess']:.1f}ms postprocess, "
                    f"{avg_times['total']:.1f}ms total")

        else:
            print("No objects detected")
            

if __name__ == '__main__':
    rospy.init_node('yolo_vit_mat_detector', anonymous=True)
    detector = YoloVitMaterialDetector()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()