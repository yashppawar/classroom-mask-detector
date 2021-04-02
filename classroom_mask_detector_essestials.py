import numpy as np
import cv2
import os
import warnings
import face_recognition

# Imports for sending the mail
import smtplib
from email.message import EmailMessage

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
    """Will return the binary version of the image"""
    temp_file_path = f"./Data/temp.{ext}"
    cv2.imwrite(temp_file_path, img)
    
    with open(temp_file_path, 'rb') as image:
        content = image.read()
    
    os.remove(temp_file_path)  # delete the temporary file

    return content  # the binary version of the image



class KnownFaceEncodings:
    """
    KnownFaceEncodings class will help in managing the encodings of the Known faces
    """
    # Constants / static members
    DATA_FOLDER = './Data/'
    DATA_FILE = 'known-faces.npy'


    def __init__(self, new_image_added:bool = False):
        KNOWN_FACE_FOLDER = KnownFaceEncodings.DATA_FOLDER + 'Known-Faces/'
        self.data = []
        
        if (KnownFaceEncodings.DATA_FILE in os.listdir(KnownFaceEncodings.DATA_FOLDER)) and not new_image_added:  
            # If there are no changes in the data return the data collected in the last data collection run
            self.data = self.load_data_file() 

        else:
            face_images = os.listdir(KNOWN_FACE_FOLDER)  # Get every face image names
            
            for image_name in face_images:  # loop through every images in the KNOWN_FACE_FOLDER
                self.__add_image(KNOWN_FACE_FOLDER + image_name, str(image_name.split('.')[0]).title())
                
            self.save_data()  # Save the data

    def __add_image(self, image_path, name):
        """Adds the given image to the data"""
        image = face_recognition.load_image_file(image_path)  # Load the image
        encodings = face_recognition.face_encodings(image)[0]  # Encode the faces in it and store the first encoded face
        self.data.append({'encodings': encodings, "name": name})

    def save_data(self):
        """Saves the collected data in self.data in the provided data file"""
        self.data = np.array(self.data)  # convert the list to np.array
        np.save(KnownFaceEncodings.DATA_FOLDER + KnownFaceEncodings.DATA_FILE, self.data)  # save the data 

    def load_data_file(self):
        """returns the np.array of the data stored in provided data file"""
        return np.load(KnownFaceEncodings.DATA_FOLDER + KnownFaceEncodings.DATA_FILE, allow_pickle=True)

    @property
    def encodings(self):
        """The numpy.array of all the encodings"""
        return np.array(list(map(lambda encoding_dict: encoding_dict["encodings"], self.data)))

    @property
    def names(self):
        """The numpy.array of all the names"""
        return np.array(list(map(lambda encoding_dict: encoding_dict["name"], self.data)))


class ReportAuthority:
    """Send Mail to the authority using this class, easy to use and will help in managing the mail"""
    def __init__(self, my_email, password, authority_email, name='AI app'):
        if my_email is None or password is None or authority_email is None:
            print('[Warning] Credentials are missing please fill them out!')
        elif 'gmail' not in my_email:
            print('[Warning] As your host is not gmail, please provide a host while sending the mail!')
        
        # Set the credentials
        self.mail_id = my_email  
        self.password = password
        self.authority_mail_id = authority_email 
        self.name = name

    def create_email(self, subject):
        self.email = EmailMessage()
        self.email['to'] = self.authority_mail_id
        self.email['Subject'] = subject
        self.email['from'] = self.name

    def set_content(self, content):
        self.email.set_content(content)

    def attach_image(self, image, image_type, name='student'):
        self.email.add_attachment(image, maintype='image', subtype=image_type, filename=f'{name}.{image_type}')

    def send(self, host='smtp.gmail.com', port=587):
        with smtplib.SMTP(host=host, port=port) as smtp:
            smtp.ehlo()
            smtp.starttls()
            
            try:
                smtp.login(self.mail_id, self.password)
                smtp.send_message(self.email)
                return True
            except:
                print("There was a problem while login try again or check if this program is allowed to use the mail services")
                return False


class WithoutMaskFace:
    """A class which will deal with the students who are not wearing a mask"""
    faces = []
    unidentified_faces = 0

    def __init__(self, name, img, cm=False):
        self.name = name
        self.image = img

        if not cm:  # I a class method is not calling it then do the following
            self.identified = True
            
            if not self._is_in_faces():  # if the person is not in faces, add him/her into it
                WithoutMaskFace.faces.append(self)

    def __repr__(self):
        return '<%s\'s Face>' % self.name 

    @classmethod
    def unidentified(cls, img, encodings):
        """Add the student who is not wearing a mask and the AI is unable to identify his/her name"""

        name = 'Undefined-' + str(WithoutMaskFace.unidentified_faces)
        obj = WithoutMaskFace(name, img, cm=True)  # using the class method

        obj.encodings = encodings
        obj.identified = False  # as it is undentified

        if not obj._is_in_faces():
            WithoutMaskFace.faces.append(obj)
            WithoutMaskFace.unidentified_faces += 1  # if a new unidentified face is added increment the variable

    def _is_in_faces(self):
        """Private member, will return `True` if the person in already identified else will return `False`."""
        if self.identified:  # if it is an identified person
            return self.__identified_is_in_face()

        else:  # If it is unidentified person check for the 
            return self.__unidentified_is_in_face()

    def __identified_is_in_face(self):
        for face in WithoutMaskFace.faces: 
            if self.name == face.name:  # if the person is found return true
                return True
            
        return False

    def __unidentified_is_in_face(self):
        if WithoutMaskFace.unidentified_faces == 0:  # if there are no faces in the faces list then return `False`
            return False
        
        matches, distance = self.__get_matches_and_distance()

        min_index = np.argmin(distance)
        
        return matches[min_index]  # return if the face is a match or not

    def __get_matches_and_distance(self):
        """Return the matches and the distance of the face encodings with the other unidentified faces"""
        matches = face_recognition.compare_faces(WithoutMaskFace.get_encodings(), self.encodings)
        distance = face_recognition.face_distance(WithoutMaskFace.get_encodings(), self.encodings)

        return matches, distance

    @staticmethod
    def flush():
        """Remove all the faces from the faces list"""
        WithoutMaskFace.faces = []

    @staticmethod
    def get_encodings():
        """"Get the encodigns of all the unidentified faces"""
        unidentified = filter(lambda obj: not obj.identified, WithoutMaskFace.faces)
        return np.array(list(map(lambda face: face.encodings[0], unidentified)))

    @staticmethod
    def report_to_authority(my_email, password:str, authority_email, name_separator:str = ',\n'):
        """
        Will send a mail to the authority, if there are students who are not wearing a mask
        will add the names of all the students in the content, the their images
        for the students whose names are unidentified, only their images will be sent.
        """
        if len(WithoutMaskFace.faces) < 1:  # If there is no one to report print that
            print('[WithoutMaskFace] No Students to report')
            return

        # create the mail or ReportAuthority object
        mail = ReportAuthority(my_email, password, authority_email, name="Student's Mask Detector AI")
        mail.create_email('Few students are not wearing a mask')

        names = WithoutMaskFace.get_names(name_separator)  
        
        content = f"Greetings,\nThe names of the students who are not wearing mask are:\n{names} "

        # if there are unidentified face tell the authority that there are some unidentified faces
        if WithoutMaskFace.unidentified_faces > 0:
            content += "\nThere are some faces also which I could not Identify. So, I have just attached their images."

        mail.set_content(content)  # add the content

        # add all the images
        for student in WithoutMaskFace.faces:
            image = get_binary(student.image)
            mail.attach_image(image, 'png', student.name)

        success = mail.send()  # Send the mail
        # Print the success/failure message
        print('[WithoutMaskFace] ' + ('Sent the Report' if success else 'Some error occured, could not send the mail!'))

    @staticmethod
    def get_names(seperator:str = ', ') -> str:
        """Returns the string of the all names combined"""
        identified_faces = filter(lambda obj: obj.identified, WithoutMaskFace.faces)  # get all the instances of identified face object
        return seperator.join(list(map(lambda student: student.name, identified_faces)))
