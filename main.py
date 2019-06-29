import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
# list of labels 
subjects = ["", "Taylor Hill", "David Beckham", "Okamoto Natsumi"]

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     
    face_cascade = cv2.CascadeClassifier('opencv-files/haarcascade_frontalface_default.xml')   
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
    
    if (len(faces) == 0):
        return None, None
    
    # assumption --> only one face,
    #extract the face area
    (x, y, w, h) = faces[0]
    # face part of the image
    return gray[y:y+w, x:x+h], faces[0]


def prepare_training_data(data_folder_path):
    dirs = os.listdir(data_folder_path)    
    faces = []
    labels = []  
    
    for dir_name in dirs:
        if not dir_name.startswith("s"):
            continue;
            
        label = int(dir_name.replace("s", ""))
        subject_dir_path = data_folder_path + "/" + dir_name  
        subject_images_names = os.listdir(subject_dir_path)
        
    
        #detect face and add face to list of faces
        for image_name in subject_images_names:           

            image_path = subject_dir_path + "/" + image_name
            image = cv2.imread(image_path)
            
            #display images to train
            print("Training on image...")
            
            #detect face
            face, rect = detect_face(image)           
          
            # ignore all faces that are not detected
            if face is not None:
                faces.append(face)
                labels.append(label)
            
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    
    return faces, labels


print("Lets gather some data")
faces, labels = prepare_training_data("training-data")
print("Total data to train: ", len(faces))
# we will use Local Binary Patterns Histograms recognizer to classify the test image
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
# or you can play with some other available recognizers as well
#face_recognizer = cv2.face.EigenFaceRecognizer_create()
#face_recognizer = cv2.face.FisherFaceRecognizer_create()

#training starts here..........
face_recognizer.train(faces, np.array(labels))

w=0
h=0
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
def draw_text(img, text, x, y,confidence):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), int(1.5))    
    cv2.putText(img, str(confidence), (x+w,y+h+100), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)


def predict(test_img):    
    face, rect = detect_face(test_img)
    label, confidence = face_recognizer.predict(face)
    ## calculating accuracy 
    if (confidence < 100):
        label = subjects[label]
        confidence = "  {0}%".format((round(confidence)))
        
    else:
        label = subjects[label]
        confidence = "  {0}%".format(abs(round(100 - confidence)))
        
#get name of respective label returned by face recognizer
    
    draw_text(test_img, label, rect[0], rect[1]-5,confidence)
    draw_rectangle(test_img, rect)    
    return test_img


print("predicting images...")

#load test images
test_img1 = cv2.imread("test-data/taylor_hill.jpg")
test_img2 = cv2.imread("test-data/dbeck.jpg")
test_img3 = cv2.imread("test-data/okamoto-natsumi.jpg")
#perform a prediction
predicted_img1 = predict(test_img1)
predicted_img2 = predict(test_img2)
predicted_img3 = predict(test_img3)

#Write ảnh đã dự đoán ra file .jpg và hiển thị trên console
cv2.imwrite('predicted_img1.jpg',predicted_img1)
plt.imshow(cv2.cvtColor(predicted_img1,cv2.COLOR_BGR2RGB))
plt.show()

cv2.imwrite('predicted_img2.jpg',predicted_img2)
plt.imshow(cv2.cvtColor(predicted_img2,cv2.COLOR_BGR2RGB))
plt.show()

cv2.imwrite('predicted_img3.jpg',predicted_img3)
plt.imshow(cv2.cvtColor(predicted_img3,cv2.COLOR_BGR2RGB))
plt.show()




