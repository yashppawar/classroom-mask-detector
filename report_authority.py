import smtplib
from email.message import EmailMessage


class ReportAuthority:
    """
    Create an email to report to authority
    """
    def __init__(self, my_email, password, authority_email, name='AI app'):
        if my_email is None or password is None or authority_email is None:
            print('\033[93m Credentials are missing please fill them out! \033[0m')
        
        if 'gmail' not in my_email: print('\033[93m As your host is not gmail, please provide a host while sending the mail! \033[0m')
        
        self.mail_id = my_email
        self.password = password
        self.authority_mail_id = authority_email 
        self.name = name

    def create_email(self, subject):
        """
        Instantiate the mail.
        """
        self.email = EmailMessage()
        self.email['to'] = self.authority_mail_id
        self.email['Subject'] = subject
        self.email['from'] = self.name

    def set_content(self, content):
        """
        Set the content for the email
        """
        self.email.set_content(content)

    def attach_image(self, image, image_type, name='student'):
        """
        Attach the given image to the email.
        """
        self.email.add_attachment(image, maintype='image', subtype=image_type, filename=f'{name}.{image_type}')

    def send(self, host='smtp.gmail.com', port=587):
        """
        Send the mail created.
        """
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
