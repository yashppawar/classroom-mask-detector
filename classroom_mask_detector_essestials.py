import numpy as np
import cv2
import os
import face_recognition

# Imports for sending the mail
from report_authority import ReportAuthority

# Constants
IMAGE_SIZE = 32  # Image size 32 x 32 px
THRESHOLD = 0.90  # only if the model is sure more than 90% then only give the prediction
WITHOUT_MASK = 1
RUN_FOR = 10 # Run for 10 seconds


# Functions
def preprocess_image(image):
    """
    preprocess_image will preprocess the image (convert it to GRAY, increase the contrast and make all the vaues float)
    :param image: The image on which preprocessing is to be done
    :return image: The preprocessed version of the image
    """
    image = image.astype("uint8")  # Make the image into int type
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert the image to Gray
    image = cv2.equalizeHist(image)  # Improve the contrast of the image
    image = image / 255  # Make all the values between 0 and 1
    
    return image


# Function to convert the image into binary
def get_binary(img, ext='png'):
    """
    Gets the binary of the image and returns it. 
    """
    temp_file_path = f"./Data/temp.{ext}"
    cv2.imwrite(temp_file_path, img)
    
    with open(temp_file_path, 'rb') as image:
        content = image.read()
    
    os.remove(temp_file_path)  # delete the temporary file

    return content  # the binary version of the image


class KnownFaceEncodings:
    """
    KnownFaceEncondings class will help in managing the encodings of the Known faces
    """
    DATA_FOLDER = './Data/'
    DATA_FILE = 'known-faces.npy'


    def __init__(self, new_image_added:bool = False):
        images = []  # for temporary storage
        KNOWN_FACE_FOLDER = self.DATA_FOLDER + 'Known-Faces/'
        
        if (self.DATA_FILE in os.listdir(self.DATA_FOLDER)) and not new_image_added:  
            # If there are no changes in the data return the data collected in the last data collection run
            self.data = np.load(self.DATA_FOLDER + self.DATA_FILE, allow_pickle=True) 

        else:
            face_images = os.listdir(KNOWN_FACE_FOLDER)  # Get every face image names
            
            for image_name in face_images:  # loop through every images in the KNOWN_FACE_FOLDER
                image = face_recognition.load_image_file(KNOWN_FACE_FOLDER + image_name)
            
                encodings = face_recognition.face_encodings(image)[0]  # Encode the faces in it and store the first encoded face
                images.append({'encodings': encodings, "name":str(image_name.split('.')[0]).title()})
                
            self.data = np.array(images)
            np.save(self.DATA_FOLDER + self.DATA_FILE, self.data)

    @property
    def encodings(self):
        """The numpy.array of all the encodings"""
        return np.array(list(map(lambda encoding_dict: encoding_dict["encodings"], self.data)))

    @property
    def names(self):
        """The numpy.array of all the names"""
        return np.array(list(map(lambda encoding_dict: encoding_dict["name"], self.data)))

class IdentifiedFace:
    """A Class to handel the identified faces which are not wearing mask"""
    faces = []
    def __init__(self, name, img):
        self.name = name
        self.image = img
        
        if not self.__is_in_faces():  # if the person is not in faces, add him/her into it
            IdentifiedFace.faces.append(self)
        
    def __is_in_faces(self):
        """Private, will return `True` if it is in IdentifiedFace.faces else `False`"""
        for face in IdentifiedFace.faces:
            if self.name == face.name:
                return True
            
        return False

    @staticmethod
    def flush():
        IdentifiedFace.faces = []

    @staticmethod
    def report_to_authority(my_email, password, authority_email):
        if len(IdentifiedFace.faces) < 1:
            print("[IdentifiedFace] No students to report")
            return 

        mail = ReportAuthority(my_email, password, authority_email, name="Student's Mask Detector AI")
        mail.create_email('Few Students not wearing a Mask')

        names = ',\n'.join(list(map(lambda student: student.name, IdentifiedFace.faces)))
        content = "Greetings,\nThe names of the student who are not weraing mask are:\n" + names
        mail.set_content(content)

        for student in IdentifiedFace.faces:
            image = get_binary(student.image)
            mail.attach_image(image, 'png', student.name)

        success = mail.send()
        print('[IdentifiedFace] ' + ('Sent the Report' if success else 'Some error occured, could not send the mail!'))



class UnidentifiedFace:
    faces = []
    def __init__(self, img, encodings):
        self.encodings = encodings
        self.image = img
        self.name = 'Unidentified-' + str(len(UnidentifiedFace.faces))

        if not self.__is_in_face():
            UnidentifiedFace.faces.append(self)     

    def __is_in_face(self):
        if len(UnidentifiedFace.faces) == 0:
            return False
        
        matches = face_recognition.compare_faces(UnidentifiedFace.get_encodings(), self.encodings)
        distance = face_recognition.face_distance(UnidentifiedFace.get_encodings(), self.encodings)

        min_index = np.argmin(distance)
        if matches[min_index]:
            return True
        return False

    @staticmethod
    def get_encodings():
        return np.array(list(map(lambda face: face.encodings[0], UnidentifiedFace.faces)))

    @staticmethod
    def flush():
        UnidentifiedFace.faces = []

    @staticmethod
    def report_to_authority(my_email, password, authority_email):
        if len(UnidentifiedFace.faces) < 1:
            print("[UnidentifiedFace] No unidentified students to report")
            return 

        mail = ReportAuthority(my_email, password, authority_email, name="Student's Mask Detector AI")
        mail.create_email('Few Students not wearing a Mask!')

        content = "Greetings,\nI Could not Identify the names of the students but their images are attached below\n"
        mail.set_content(content)

        for student in IdentifiedFace.faces:
            image = get_binary(student.image)
            mail.attach_image(image, 'png', student.name)

        success = mail.send()
        print('[UnidentifiedFace] ' + ('Sent the Report' if success else 'Some error occured, could not send the mail!'))
