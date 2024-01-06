import os
import glob
import cv2
import mediapipe as mp
import numpy as np
import math

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

#####################################################################################
############################## image_preprocess_(size) ##############################
#####################################################################################

# face_detection + image_resize
# face_detection + image_resize to 64x64 -> image_preprocess_64
# face_detection + image_resize to 128x128 -> image_preprocess_128
# face_detection + image_resize to 256x256 -> image_preprocess_256
# face_detection + image_resize to 512x512 -> image_preprocess_512

def image_preprocess_64(file):

    image = cv2.imread(file)
    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5) as face_detection:
        image_width = image.shape[1]
        image_height = image.shape[0]
        #image = cv2.imread(file)
        # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Draw face detections of each face.
        
        annotated_image = image.copy()
        for detection in results.detections:
            # print('Nose tip:')
            # print(mp_face_detection.get_key_point(
            #     detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))
            #print(detection)
            normalized_x=detection.location_data.relative_bounding_box.xmin
            normalized_y=detection.location_data.relative_bounding_box.ymin
            normalized_w=detection.location_data.relative_bounding_box.width
            normalized_h=detection.location_data.relative_bounding_box.height
            x = min(math.floor(normalized_x * image_width), image_width - 1)
            y = min(math.floor(normalized_y * image_height), image_height - 1)
            w = min(math.floor(normalized_w * image_width), image_width - 1)
            h = min(math.floor(normalized_h * image_height), image_height - 1)
            print(x,y,w,h)
            #mp_drawing.draw_detection(annotated_image, detection)
            annotated_image = image[y-50:y+h, x:x+w]
            output_image = cv2.resize(annotated_image, (64, 64))             
            print("image_preprocess_64 is complete")

        f_name = (file.split("/")[-1]).split(".")[0]    
        cv2.imwrite("data/KDEF_Noside/preprocessed_" + f_name + "_64.jpg", output_image)
        print("Successfully write file")

def image_preprocess_128(file):

    image = cv2.imread(file)
    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5) as face_detection:
        image_width = image.shape[1]
        image_height = image.shape[0]
        #image = cv2.imread(file)
        # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Draw face detections of each face.
        
        annotated_image = image.copy()
        for detection in results.detections:
            # print('Nose tip:')
            # print(mp_face_detection.get_key_point(
            #     detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))
            #print(detection)
            normalized_x=detection.location_data.relative_bounding_box.xmin
            normalized_y=detection.location_data.relative_bounding_box.ymin
            normalized_w=detection.location_data.relative_bounding_box.width
            normalized_h=detection.location_data.relative_bounding_box.height
            x = min(math.floor(normalized_x * image_width), image_width - 1)
            y = min(math.floor(normalized_y * image_height), image_height - 1)
            w = min(math.floor(normalized_w * image_width), image_width - 1)
            h = min(math.floor(normalized_h * image_height), image_height - 1)
            print(x,y,w,h)
            #mp_drawing.draw_detection(annotated_image, detection)
            annotated_image = image[y-50:y+h, x:x+w]
            output_image = cv2.resize(annotated_image, (128, 128))             
            print("image_preprocess_128 is complete")

        f_name = (file.split("/")[-1]).split(".")[0]    
        cv2.imwrite("data/KDEF_Noside/preprocessed_" + f_name + "_128.jpg", output_image)
        print("Successfully write file")

def image_preprocess_256(file):

    image = cv2.imread(file)
    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5) as face_detection:
        image_width = image.shape[1]
        image_height = image.shape[0]
        #image = cv2.imread(file)
        # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Draw face detections of each face.
        
        annotated_image = image.copy()
        for detection in results.detections:
            # print('Nose tip:')
            # print(mp_face_detection.get_key_point(
            #     detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))
            #print(detection)
            normalized_x=detection.location_data.relative_bounding_box.xmin
            normalized_y=detection.location_data.relative_bounding_box.ymin
            normalized_w=detection.location_data.relative_bounding_box.width
            normalized_h=detection.location_data.relative_bounding_box.height
            x = min(math.floor(normalized_x * image_width), image_width - 1)
            y = min(math.floor(normalized_y * image_height), image_height - 1)
            w = min(math.floor(normalized_w * image_width), image_width - 1)
            h = min(math.floor(normalized_h * image_height), image_height - 1)
            print(x,y,w,h)
            #mp_drawing.draw_detection(annotated_image, detection)
            annotated_image = image[y-50:y+h, x:x+w]
            output_image = cv2.resize(annotated_image, (256, 256))             
            print("image_preprocess_256 is complete")

        f_name = (file.split("/")[-1]).split(".")[0]    
        cv2.imwrite("stargan_new_6_leaky/data/test/" + f_name + "_256.jpg", output_image)
        print("Successfully write file")

def image_preprocess_512(file):

    image = cv2.imread(file)
    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5) as face_detection:
        image_width = image.shape[1]
        image_height = image.shape[0]
        #image = cv2.imread(file)
        # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Draw face detections of each face.
        
        annotated_image = image.copy()
        for detection in results.detections:
            # print('Nose tip:')
            # print(mp_face_detection.get_key_point(
            #     detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))
            #print(detection)
            normalized_x=detection.location_data.relative_bounding_box.xmin
            normalized_y=detection.location_data.relative_bounding_box.ymin
            normalized_w=detection.location_data.relative_bounding_box.width
            normalized_h=detection.location_data.relative_bounding_box.height
            x = min(math.floor(normalized_x * image_width), image_width - 1)
            y = min(math.floor(normalized_y * image_height), image_height - 1)
            w = min(math.floor(normalized_w * image_width), image_width - 1)
            h = min(math.floor(normalized_h * image_height), image_height - 1)
            print(x,y,w,h)
            #mp_drawing.draw_detection(annotated_image, detection)
            annotated_image = image[y-50:y+h, x:x+w]
            output_image = cv2.resize(annotated_image, (512, 512))             
            print("image_preprocess_512 is complete")

        f_name = (file.split("/")[-1]).split(".")[0]    
        cv2.imwrite("data/KDEF_Noside/preprocessed_" + f_name + "_512.jpg", output_image)
        print("Successfully write file")

def image_list_preprocess_256(path):
    
    file_list = glob.glob(path)
    
    for i, file in enumerate(file_list):
        image = cv2.imread(file)
        name = file[-1][:-4]
        #print(file)
        with mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5) as face_detection:
            image_width = image.shape[1]
            image_height = image.shape[0]
            #image = cv2.imread(file)
            # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
            results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Draw face detections of each face.
            
            try :
                for detection in results.detections:
                    # print('Nose tip:')
                    # print(mp_face_detection.get_key_point(
                    #     detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))
                    #print(detection)
                    normalized_x=detection.location_data.relative_bounding_box.xmin
                    normalized_y=detection.location_data.relative_bounding_box.ymin
                    normalized_w=detection.location_data.relative_bounding_box.width
                    normalized_h=detection.location_data.relative_bounding_box.height
                    x = min(math.floor(normalized_x * image_width), image_width - 1)
                    y = min(math.floor(normalized_y * image_height), image_height - 1)
                    w = min(math.floor(normalized_w * image_width), image_width - 1)
                    h = min(math.floor(normalized_h * image_height), image_height - 1)
                    print(x,y,w,h)
                    #mp_drawing.draw_detection(annotated_image, detection)

                    annotated_image = image[y-50:y+h, x:x+w]
                    output_image = cv2.resize(annotated_image, (256, 256))             

                cv2.imwrite("stargan_new_6_leaky/data/test/{}.jpg".format(name), output_image)
                print("Saved image")
            except:
                print("error")
                pass
    print("Successfully write file")
#####################################################################################
#####################################################################################
#####################################################################################



#################################################################################
############################## image_resize_(size) ##############################
#################################################################################
# Just image_resize
# image_resize to 64x64 -> image_resize_64
# image_resize to 128x128 -> image_resize_128
# image_resize to 256x256 -> image_resize_256
# image_resize to 512x512 -> image_resize_512

def image_resize_64(file):
    image = cv2.imread(file)    
    output_image = cv2.resize(image, (64, 64))
    print("image_resize_64 is complete")
    f_name = (file.split("/")[-1]).split(".")[0]    
    cv2.imwrite("data/KDEF_Noside/preprocessed_" + f_name + "_64.jpg", output_image)
    print("Successfully write file")

def image_resize_128(file):
    image = cv2.imread(file)    
    output_image = cv2.resize(image, (128, 128))
    print("image_resize_128 is complete")
    f_name = (file.split("/")[-1]).split(".")[0]    
    cv2.imwrite("data/KDEF_Noside/preprocessed_" + f_name + "_128.jpg", output_image)
    print("Successfully write file")

def image_resize_256(file):
    image = cv2.imread(file)    
    output_image = cv2.resize(image, (256, 256))
    #print("image_resize_256 is complete")
    f_name = (file.split("/")[-1]).split(".")[0]    
    cv2.imwrite("CIFAR10" + f_name + ".jpg", output_image)
    #print("Successfully write file")

def image_resize_512(file):
    image = cv2.imread(file)    
    output_image = cv2.resize(image, (512, 512))
    print("image_resize_512 is complete")
    f_name = (file.split("/")[-1]).split(".")[0]    
    cv2.imwrite("data/KDEF_Noside/preprocessed_" + f_name + "_512.jpg", output_image)
    print("Successfully write file")
       
#################################################################################
#################################################################################
#################################################################################

def train_img_list_resize_256(file_list, name):
    print(os.getcwd())
    for i, file in enumerate(file_list):
        image = cv2.imread(file)
        #print(file)
        with mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5) as face_detection:
            image_width = image.shape[1]
            image_height = image.shape[0]
            #image = cv2.imread(file)
            # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
            results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Draw face detections of each face.
            
            try :
                for detection in results.detections:
                    # print('Nose tip:')
                    # print(mp_face_detection.get_key_point(
                    #     detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))
                    #print(detection)
                    normalized_x=detection.location_data.relative_bounding_box.xmin
                    normalized_y=detection.location_data.relative_bounding_box.ymin
                    normalized_w=detection.location_data.relative_bounding_box.width
                    normalized_h=detection.location_data.relative_bounding_box.height
                    x = min(math.floor(normalized_x * image_width), image_width - 1)
                    y = min(math.floor(normalized_y * image_height), image_height - 1)
                    w = min(math.floor(normalized_w * image_width), image_width - 1)
                    h = min(math.floor(normalized_h * image_height), image_height - 1)
                    print(x,y,w,h)
                    #mp_drawing.draw_detection(annotated_image, detection)
                    if name == 'surprised' or name == 'fearful':
                        annotated_image = image[y-50:y+h, x:x+w]
                    else:
                        annotated_image = image[y-50:y+h, x:x+w]
                    output_image = cv2.resize(annotated_image, (256, 256))             

                cv2.imwrite("data/new_reduced/{}/{}.jpg".format(name, i), output_image)
            except:
                print("error")
                pass
    print("Successfully write file")

#################################################################################
################################ main_part ######################################
#################################################################################

# file = "/home/mineslab-ubuntu/stargan/Original_jpg/robot3.jpg"
train_data_path = "/home/mineslab-ubuntu/stargan/Korean/reduced_Korean"
# #image_resize_256(file)
sad = glob.glob(os.path.join(train_data_path, 'sad') + '/*')
angry = glob.glob(os.path.join(train_data_path, 'angry') + '/*')
happy = glob.glob(os.path.join(train_data_path, 'happy') + '/*')
neutral = glob.glob(os.path.join(train_data_path, 'neutral') + '/*')
surprised = glob.glob(os.path.join(train_data_path, 'surprised') +"/*")
fearful = glob.glob(os.path.join(train_data_path, 'fearful') + '/*')


# train_img_list_resize_256(sad, 'sad')
# train_img_list_resize_256(angry, 'angry')
# train_img_list_resize_256(happy, 'happy')
# train_img_list_resize_256(neutral, 'neutral')
# train_img_list_resize_256(surprised, 'surprised')
# train_img_list_resize_256(fearful, 'fearful')

# print(len(sad), len(angry), len(happy), len(neutral))
# f_name = os.getcwd() + "/data/cartoon/"
# fearful = glob.glob(f_name)
# image_list_preprocess_256(f_name)

# file = "/home/mineslab-ubuntu/stargan/Original_jpg/sakura_ani2.png"
# image_resize_256(file)

#bts = "/home/mineslab-ubuntu/stargan/Original_jpg/sakura_ani2.png"
#img_path = "/home/mineslab-ubuntu/stargan/Original_jpg"
img_file = glob.glob("/home/mineslab-ubuntu/stargan/CIFAR10/*")
for image in img_file:
    print(image)
    image_resize_256(image)
#image_list_preprocess_256(img_path)
# image_resize_64(img_path)
# image_resize_512(img_path)