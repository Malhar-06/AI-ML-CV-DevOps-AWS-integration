import streamlit as st # This library is used to create web interfaces.
import pyttsx3 # Text-to-speech conversion library.
import requests # This library used for making HTTP requests.
import pandas as pd # This data manipulation and analysis library.
from sklearn.linear_model import LinearRegression # scikit-learn Open-source ML library. And sklearn.linear_model is a module. And LinearRegression is the class.
from sklearn.model_selection import train_test_split # train_test_split is a function from the sklearn.model_selection module
import time # Module
import cv2 # OpenCV library
from transformers import pipeline # Lib "transformers, The pipeline function provides a high-level API for various NLP tasks such as text classification, sentiment analysis, translation, summarization, and more.
import openai # Openai Library
from aws_config import ec2
import traceback
import mediapipe as mp 


def main():

    menu_options = ["Docker Terminal","AWS_EC2_Gestures","Sentiment Analysis","CHIPITI","Automated ML","Use Goggles","Face Blur","Face Distance","Text to Speech","Python Interpreter","SMS Sender","Email Sender","Linux Terminal"]
    
    st.sidebar.title("What's your Plan?")
    choice = st.sidebar.selectbox("Select an option", menu_options)

    
    if choice == menu_options[0]:
        Docker_Terminal()
    elif choice == menu_options[1]:
        AWS_EC2_Gestures()
    elif choice == menu_options[2]:
        Sentiment_ana()
    elif choice == menu_options[3]:
        chipiti()
    elif choice == menu_options[4]:
        automl()
    elif choice == menu_options[5]:
        use_goggles()
    elif choice == menu_options[6]:
        Face_blur()
    elif choice == menu_options[7]:
        face_distance()
    elif choice == menu_options[8]:
        textTSpeech()
    elif choice == menu_options[9]:
        python_Interpreter()
    elif choice == menu_options[10]:
        sms_sender()
    elif choice == menu_options[11]:
        email_sender()   
    elif choice == menu_options[12]:
        linux_bash()

    


def Docker_Terminal():

    # Function to execute Docker commands using the CGI script
    def execute_docker_command(command):
        cgi_url = "http://13.235.2.174/cgi-bin/hello.py"
        response = requests.post(cgi_url, data={"command": command})
        return response.text



    st.title("Docker Admin")

    # Docker commands as buttons
    if st.button("List Docker Images"):
        result = execute_docker_command("docker images")
        st.subheader("Docker Images:")
        st.text(result)

    if st.button("List Docker Containers"):
        result = execute_docker_command("docker ps")
        st.subheader("Docker Containers:")
        st.text(result)

    st.subheader("Pull Docker Image")
    image_name = st.text_input("Enter the Docker image name:")
    if st.button("Pull"):
        command = f"docker pull {image_name}"
        result = execute_docker_command(command)
        st.subheader("Command Output:")
        st.text(result)
    
    st.subheader("Custom Command")
    custom_command = st.text_input("Enter Custom Docker Command:")
    if st.button("Execute Custom Command"):
        result = execute_docker_command(custom_command)
        st.subheader("Command Output:")
        st.text(result)



def AWS_EC2_Gestures():

    def get_instances_with_tag(tag_key, tag_value):
        instances = ec2.instances.filter(
            Filters=[
                {'Name': f'tag:{tag_key}', 'Values': [tag_value]},
                {'Name': 'instance-state-name', 'Values': ['pending', 'running', 'stopping', 'stopped']}
            ]
        )
        return [instance.id for instance in instances]

    def my_os_launch():
        instances = ec2.create_instances(
            ImageId="ami-0ec0e125bb6c6e8ec",
            MinCount=1,
            MaxCount=1,
            InstanceType="t2.micro",
            SecurityGroupIds=["sg-0401e4f8083ed08e8"],
            TagSpecifications=[
                {'ResourceType': 'instance', 'Tags': [{'Key': 'Name', 'Value': 'for-python-cv2'}]}
            ]
        )
        my_id = instances[0].id
        st.sidebar.write("Total Number of OS:", len(get_instances_with_tag('Name', 'for-python-cv2')))
        st.sidebar.write("Launched Instance ID:", my_id)

    def os_terminate():
        all_os = get_instances_with_tag('Name', 'for-python-cv2')
        if all_os:
            os_delete = all_os.pop()
            ec2.instances.filter(InstanceIds=[os_delete]).terminate()
            st.sidebar.write("Terminated Instance ID:", os_delete)
            st.sidebar.write("Total number of instances:", len(get_instances_with_tag('Name', 'for-python-cv2')))
        else:
            st.sidebar.write("No instances to terminate.")

    run = st.sidebar.checkbox('Run')
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
        return

    index_finger_detected = False
    thumb_detected = False
    last_index_detection_time = 0
    last_thumb_detection_time = 0
    detection_delay = 2

    while run:
        ret, photo = cap.read() # cap means Capture  and ret means return it is a boolean like if cap return something then it will become true.
        # The cap.read() function from OpenCV returns a tuple with two values :- ret and photo.
        if not ret:
            st.error("Failed to capture image from camera.")
            break

        photo_rgb = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB) # MediaPipe expect images in RGB format, so, converting them from BGR to RGB.
        results = hands.process(photo_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # It is a constant provided by the MediaPipe library, specifically within the mp_hands module. It represents the index finger tip landmark in hand tracking.
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP] # Tip of the index finger
                index_finger_dip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP] # Distal Interphalangeal Joint (DIP)
                index_finger_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP] # Proximal Interphalangeal Joint (PIP)
                index_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP] # Metacarpophalangeal Joint (MCP)

                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]# Tip of the thumb
                thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]  # Interphalangeal joint of the thumb
                thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC]# Carpometacarpal joint of the thumb


                current_time = time.time()
                
                # here x,y,z are the positons of the finger's landmark
                # x: Horizontal position, ranging from 0 to 1 (left to right across the image).
                # y: Vertical position, ranging from 0 to 1 (top to bottom across the image).
                if (index_finger_tip.y < index_finger_dip.y < index_finger_pip.y < index_finger_mcp.y) and \
                        (index_finger_tip.x > index_finger_dip.x > index_finger_pip.x > index_finger_mcp.x):
                    if not index_finger_detected and current_time - last_index_detection_time > detection_delay:
                        cv2.putText(photo, "Index Finger Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        st.sidebar.write("Index Finger Detected")
                        my_os_launch()
                        index_finger_detected = True
                        last_index_detection_time = current_time
                else:
                    index_finger_detected = False

                if (thumb_tip.y < thumb_ip.y < thumb_mcp.y) and \
                        (thumb_tip.x < thumb_ip.x < thumb_mcp.x):
                    if not thumb_detected and current_time - last_thumb_detection_time > detection_delay:
                        cv2.putText(photo, "Thumb Detected", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        st.sidebar.write("Thumb Detected")
                        os_terminate()
                        thumb_detected = True
                        last_thumb_detection_time = current_time
                else:
                    thumb_detected = False

        if photo is not None:  # Ensure photo is not None before displaying
            FRAME_WINDOW.image(photo, channels="BGR")  # Display the frame in Streamlit

    hands.close()
    cv2.destroyAllWindows()
    cap.release()



# Pipeline initializes a pre-trained sentiment analysis pipeline from the Hugging Face library.
def Sentiment_ana():

    nlp = pipeline("sentiment-analysis")
   
    st.title("Sentiment Analysis App")

    # Get user input for text
    text = st.text_area("Enter some text:")

    if text:
        # Analyze the sentiment of the text
        result = nlp(text)[0]

        # Extract sentiment label and score
        sentiment_label = result["label"]
        sentiment_score = result["score"]

        # Display the result
        st.subheader("Sentiment Analysis Result:")
        st.write(f"Sentiment: {sentiment_label.capitalize()}")
        st.write(f"Confidence Score: {sentiment_score:.2f}")

    



def chipiti():

    # Function to generate response using the GPT model
    def generate_response(prompt):
        system_prompt = """You are a knowledgeable assistant specializing in technology, 
        particularly in DevOps, cloud computing, Python, and Django. Provide concise and 
        informative answers to questions related to these topics."""

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150
        )
        return response.choices[0].message['content'].strip()

    st.title("Technology Learning Portal")
    st.write("Enter your question or topic below to learn about technologies:")
    st.write("(Focus areas: DevOps, Cloud, Python, Django)")

    user_input = st.text_area("Your Question:")
    
    if st.button("Get Answer"):
        if user_input:
            response = generate_response(user_input)
            st.subheader("Answer:")
            st.write(response)
            
            # Text-to-speech conversion
            engine = pyttsx3.init()
            engine.say(response)
            engine.runAndWait()
        else:
            st.warning("Please enter a question or topic.")
       



def automl():
    st.title("Automated Machine Learning Model")
    
    # File path input with unique key
    file = st.text_input("Enter the file path:")
    
    if not file:
        st.warning("Please enter a file path.")
        return
    
    try:
        data = pd.read_csv(file)
    except FileNotFoundError:
        st.error("File not found. Please check the file path.")
        return
    
    st.write("Columns in Dataset:", data.columns.tolist())  # Display column names for debugging
    
    # X labels input with unique key
    a = st.text_input("Enter the X labels (space-separated):")
    x_labels = a.split() # Split user input into a list of column names
    
    if not all(label in data.columns for label in x_labels): # Traverses through x_lbles column to check whether any value is missing
        st.error("One or more X labels are not found in the dataset.")
        return
    
    x = data[x_labels]  # Select columns based on user input
    
    # Y label input with unique key
    y_label = st.text_input("Enter the Y label:")
    
    if y_label not in data.columns:
        st.error(f"Column '{y_label}' not found in the dataset.")
        return
    
    y = data[y_label]
    
    if len(x_labels) == 1:
        x = x.values.reshape(-1, 1)  # Reshape for single feature case
        # To, each row represents a sample and each column represents a feature.
        # How can we change reshape func. value?
        # (1,-1) # 1:-one row , -1:-all elements
        # (2,-1) # 2:-two row , -1:-all elements
        # (3,-1) # 3:-three row , -1:-all elements
        # (-1,1) # -1:- all elements, 1:- one column
        # (-1,2) # -1:- all elements, 2:- two column
        # (-1,3) # -1:- all elements, 3:- three column
        model = LinearRegression()
        model.fit(x, y)
        st.write("Model Coefficients:", model.coef_)
    else: #When len(x_labels) > 1, x is already a 2D array (with multiple columns), so you don’t need to reshape it.
        x_train, y_train = train_test_split(x, y, test_size=0.2, random_state=3)
        model = LinearRegression()
        model.fit(x_train, y_train)
        st.write("Model Coefficients:", model.coef_)





def use_goggles():
    run = st.checkbox('Run', value=True)  # Default to checked
    FRAME_WINDOW = st.image([])  # Placeholder for the image
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') # "cv2.data.haarcascades" is a path in cv2 module where all the cascade classifiers are placed and we are put our file there.
    accessories_image = cv2.imread("goggles.png", cv2.IMREAD_UNCHANGED) # IMREAD_UNCHANGED means the image should be loaded as-is (upon the face), without any conversions.
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
        return

    while run:
        ret, frame = cap.read() #read func return 2 values ret bool and photo/frame itself (data)
        if not ret: # If ret==True then only go forward bypassing the if block
            st.error("Failed to capture frame from webcam. Exiting...")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        # The "MultiScale" in the name refers to its ability to detect faces of various sizes in the image.
        # This parameter specifies how much the image size is reduced in terms of size(memory) at each image scale. A value of 1.1 means the image is reduced by 10% at each scale.
        # If we reduce less size then detection will be clearer but computation required will be high.
        # minNeighbors parameter specifies how many neighbors each candidate rectangle should have to retain it. Each rectangle represents a face here.
        # minSize:-- 30x30 pixels, meaning faces smaller than this in the image will not be detected

        # 
        # The detectMultiScale() method returns a list of rectangles where each rectangle represents a detected face. Each rectangle is a tuple of four values: (x, y, w, h), where x and y are the top-left coordinates of the face, and w and h are its width and height.
        
        for (x, y, w, h) in faces:
            resized_sunglasses = cv2.resize(accessories_image, (w, h))
            
            # Get the alpha channel
            alpha_channel = resized_sunglasses[:,:,3]
            
            # Create a mask from the alpha channel
            mask = alpha_channel != 0
            
            # Calculate the region of interest in the frame
            roi = frame[y:y+h, x:x+w]
            
            # Apply the accessory only where the mask is True
            roi[mask] = resized_sunglasses[:,:,:3][mask]

            # Update the frame with the modified ROI
            frame[y:y+h, x:x+w] = roi
        
        # Display the frame in the Streamlit app
        FRAME_WINDOW.image(frame, channels="BGR")
        
        # Display the frame in the Streamlit app
        FRAME_WINDOW.image(frame, channels="BGR")
        
        # Optionally include a break in case the checkbox is toggled off during streaming
        if not run:
            break
    
    cap.release()
    st.write('Stopped')



def Face_blur():
    run_b = st.checkbox('Run')
    FRAME_WINDOW = st.image([])

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    video_capture = cv2.VideoCapture(0)  

    while run_b:
        _, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        # roi :- region of interest
        for (x, y, w, h) in faces:
            # y:y+h selects rows from y to y+h (not including y+h)
            # x:x+w selects columns from x to x+w (not including x+w)  
            roi = frame[y:y+h, x:x+w]
            # Gaussian blur is a widely used effect in graphics software, typically to reduce image noise and reduce detail. 
            # roi is the image we want to blur
            # 99*99 specifies the size of the Gaussian kernel used for blurring.
            # 30 specifies the size of the Gaussian kernel used for blurring.
            blurred_roi = cv2.GaussianBlur(roi, (99, 99), 30)
            frame[y:y+h, x:x+w] = blurred_roi

        FRAME_WINDOW.image(frame, channels="BGR")
        # run = st.checkbox('Run')  # Update the state of 'run' within the loop

    video_capture.release()
    st.write('Stopped')




def face_distance():
    run_d = st.checkbox('Run')
    FRAME_WINDOW = st.image([])        
    KNOWN_FACE_WIDTH = 15.0  
    FOCAL_LENGTH = 560.0     

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    video_capture = cv2.VideoCapture(0)  

    while run_d:
        _, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # (0, 255, 0):This tuple represents the color of the rectangle in BGR format (Blue, Green, Red).    and 2 is thickness of the rectangle.
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            face_width_pixels = w
            distance = (KNOWN_FACE_WIDTH * FOCAL_LENGTH) / face_width_pixels
            # we assumed face width that's why the results we are getting are sightly inaccurate face distance.
            # Formula works like as we get our face away from camera, the pixels of the face get decrease and that's why distance of the face increases.


            distance_text = f"Distance: {distance:.2f} cm"
            cv2.putText(frame, distance_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # y-10 places the text 10 pixels above the top of the face rectangle.
            # 0.7 text font scale factor. larger value larger text.

        FRAME_WINDOW.image(frame, channels="BGR")
        # run = st.checkbox('Run')  # Update the state of 'run' within the loop

    video_capture.release()
    st.write('Stopped')




def textTSpeech():
    st.header("Convert Text into Speech")
    # Add your code for Option 3 here
    message = st.text_input("Enter message to speak:")
    if st.button("Speak"):
        if message:
            myspeaker = pyttsx3.init() # initializes the pyttsx3 text-to-speech engine
            myspeaker.say(message)
            myspeaker.runAndWait()
            st.success("Text converted to speech!")



def python_Interpreter():
   
    def execute_code(code):
        # Capture standard output
        from io import StringIO
        import sys
        old_stdout = sys.stdout
        redirected_output = sys.stdout = StringIO()
        # StringIO() object can be used as input or output to the most function like stdout,stdin,stderr that would expect a standard file object.


        try:
            exec(code)
        except Exception:
            st.error(traceback.format_exc())

        # Reset standard output
        sys.stdout = old_stdout

        return redirected_output.getvalue()

    st.title("Python Interpreter")

    # Text area for user input
    code = st.text_area("Enter your Python code below:", height=300)

    # Execute button
    if st.button("Execute"):
        if code.strip():
            output = execute_code(code)
            st.code(output, language="python")
        else:
            st.warning("Please enter some Python code.")





def sms_sender():
    from twilio.rest import Client

    # Replace these with your Twilio credentials
    TWILIO_ACCOUNT_SID = 'your_twilio_a/c_SID'
    TWILIO_AUTH_TOKEN = 'your_twilio_a/c_auth_token'
    TWILIO_PHONE_NUMBER = 'your_twilio_a/c_number'

    def send_sms(phone_number, message):
        try:
            client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            client.messages.create(
                to=phone_number,
                from_=TWILIO_PHONE_NUMBER,
                body=message
            )
            return True
        except Exception as e:
            st.error(f"Failed to send SMS. Error: {str(e)}")
            return False
 
    st.title("Automatic Text SMS Sender")

    phone_number = st.text_input("Enter recipient's phone number:")
    message = st.text_area("Enter your message:")

    if st.button("Send SMS"):
        if phone_number and message:
            st.info("Sending SMS...")
            if send_sms(phone_number, message):
                st.success("SMS sent successfully!")
            else:
                st.error("Failed to send SMS. Please check the phone number and Twilio credentials.")
        else:
            st.warning("Please enter a valid phone number and message.")

  



def email_sender():
    import smtplib
    from email.mime.text import MIMEText # Used to create the structure of the email (text and multipart).
    from email.mime.multipart import MIMEMultipart
    import streamlit as st

    # Gmail credentials
    GMAIL_EMAIL = 'abc@gmail.com'
    GMAIL_PASSWORD = 'hjkjfrdrnbvnfhj'  # Use the app password generated from Google.

    def send_email(subject, recipients, message):
        try:
            msg = MIMEMultipart()
            msg['From'] = GMAIL_EMAIL
            msg['To'] = ", ".join(recipients)
            msg['Subject'] = subject

            body = message
            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP('smtp.gmail.com', 587) #Port 587, combined with STARTTLS(transfer layer security). ensures encrypted connection betwn the client and email server.
            server.starttls()
            server.login(GMAIL_EMAIL, GMAIL_PASSWORD)
            server.sendmail(GMAIL_EMAIL, recipients, msg.as_string())
            server.quit()

            return True
        except Exception as e:
            st.error(f"Failed to send email. Error: {str(e)}")
            return False


    st.title("Automatic Email Sender")

    recipients = st.text_input("Enter recipient(s) email address (comma-separated):")
    subject = st.text_input("Enter email subject:")
    message = st.text_area("Enter your message:")

    if st.button("Send Email"):
        if recipients and subject and message:
            recipient_list = [email.strip() for email in recipients.split(',')]
            st.info("Sending Email...")
            if send_email(subject, recipient_list, message):
                st.success("Email sent successfully!")
            else:
                st.error("Failed to send email. Please check your Gmail credentials.")
        else:
            st.warning("Please enter valid recipient email address(es), subject, and message.")

  



def linux_bash():
    # Function to execute Linux commands using the CGI script
    def execute_linux_command(command):
        cgi_url = "http://13.235.2.174/cgi-bin/hello.py"
        response = requests.post(cgi_url, data={"command": command})
        return response.text

    # Streamlit app

    st.title("Linux Terminal")

    # Sidebar with options
    st.sidebar.subheader("Options")
    command_options = ["ls", "pwd", "date", "ifconfig", "Custom Command"]
    selected_option = st.sidebar.selectbox("Select a command:", command_options)

    if selected_option == "Custom Command":
        command = st.text_input("Enter a Linux command:")
    else:
        command = selected_option

    if st.sidebar.button("Execute"):
        result = execute_linux_command(command)
        st.subheader("Command Output:")
        st.code(result, language="shell")



if __name__ == "__main__":
    main()

