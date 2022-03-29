# Import standard dependencies
import cv2
import os
import numpy as np

from tensorflow.keras.layers import Layer
import tensorflow as tf
import uuid

def face_crop(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,4)
    face = image
    face_dected = False
    for (x,y,w,h) in faces:
        face_dected = True
        face = image[y:y+h,x:x+w]
        face = cv2.resize(face, (250,250))
        cv2.imshow("face",face)

    return face_dected,face

def preprocess(file_path):
    # Read in image from file path
    byte_img = tf.io.read_file(file_path)
    # Load in the image
    img = tf.io.decode_jpeg(byte_img)

    # Preprocessing steps - resizing the image to be 100x100x3
    img = tf.image.resize(img, (100, 100))
    # Scale image to be between 0 and 1
    img = img / 255.0

    # Return image
    return img


# Siamese L1 Distance class
class L1Dist(Layer):

    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)




def verify(model, detection_threshold, verification_threshold):
    input_img = preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
    #print("Before Outer Loop")
    for folder_name in os.listdir("application_data"):
        if folder_name == "input_image":
            continue
        
    # Build results array
    #Finiding total employee count
    #Walking through each directory and checking image of that directory to input image
        results = []
        #print("Before Inner Loop")
        for image in os.listdir(os.path.join('application_data', folder_name)):
            validation_img = preprocess(os.path.join('application_data', folder_name, image))

            # Make Predictions
            result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
            results.append(result)
        # Detection Threshold: Metric above which a prediciton is considered positive
        detection = np.sum(np.array(results) > detection_threshold)
        # Verification Threshold: Proportion of positive predictions / total positive samples
        verification = detection / len(os.listdir(os.path.join('application_data', folder_name)))
        # print("Checking to verify")

        if verification > verification_threshold:
            verified = True
            print("Returning values")
            print("Folder name: ", folder_name)
            print("result : ", results)
            return results, verified, folder_name
        print("Folder name: ",folder_name)
        print("result : ",results)


    return results, False, "Not found"


def add_person(name_id):

    #path to main folder + individual person photo folder
    path = 'D:\\path\\to\\main\\folder\\' + name_id  #path to main folder + individual person photo folder
    print(path)
    if(os.path.isdir(path)):
        print("alreay in database")
        return
    else:
        os.makedirs(path)


    # Establish a connection to the webcam
    cap = cv2.VideoCapture(0)
    i=1
    while cap.isOpened():
        ret, frame = cap.read()
        if(i>50):
            break

        cv2.imshow('Image Collection', frame)
        # Cut down frame to 250x250px
        face_detect, frame = face_crop(frame)
        print(face_detect)
        # Collect anchors
        if cv2.waitKey(1) & 0XFF == ord('a'):
            if(face_detect):
                # Create the unique file path
                imgname = os.path.join(path, '{}.jpg'.format(uuid.uuid1()))
                # Write out anchor image
                cv2.imwrite(imgname, frame)
                i = i + 1

        # Breaking gracefully
        if cv2.waitKey(1) & 0XFF == ord('q'):
            break

    # Release the webcam
    cap.release()
    # Close the image show frame
    cv2.destroyAllWindows()

def verify_a_person():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow('Verification', frame)
        face_detect,frame = face_crop(frame)
        if(not face_detect):
            continue


        # Verification trigger
        if cv2.waitKey(10) & 0xFF == ord('v'):

            cv2.imwrite(os.path.join('application_data', 'input_image', 'input_image.jpg'), frame)
            # Run verification
            results, verified, folder_name = verify(siamese_model, 0.4, 0.4)
            print(verified)
            # print(results)
            print(folder_name)


        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Reload model
siamese_model = tf.keras.models.load_model('siamesemodelv2.h5',
                                           custom_objects={'L1Dist': L1Dist, 'BinaryCrossentropy': tf.losses.BinaryCrossentropy})

while(1):

    choice = input("What do you want to do : \n1.Add a new person\n2.Verify person\n3.Quit\n")
    global face_cascade

    #path to haar cascade file
    haar_file = 'D:\\path\\to\\haar_cascade_file\\haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(haar_file)
    if(choice == '1'):
        name_id = input("Enter person's identification: ")
        add_person(name_id)
    elif(choice == '2'):
        verify_a_person()
    else:
        print(choice)
        print("Thankyou")
        break
