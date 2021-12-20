import csv 
import boto3
import cv2
import numpy as np
from colorama import Fore, Back, Style
print(Fore.BLUE + "\n\nWELCOME TO AWS REKOGNITION APPLICATION\n\nTHE FEATURES THAT ARE AVAILABLE IN THIS APPLICATION ARE MENTIONED BELOW\n ")
print(Style.RESET_ALL)

print(Fore.RED+"SELECT ANY OF THE FEATURE YOU WANT TO USE\n"+Style.RESET_ALL)
print(Fore.BLACK+Back.CYAN+"1. OBJECT AND SCENE DETECTION"+Style.RESET_ALL)
print("\n")
print(Fore.BLACK+Back.CYAN+"2. IMAGE MODERATION"+Style.RESET_ALL)
print("\n")
print(Fore.BLACK+Back.CYAN+"3. FACIAL ANALYSIS"+Style.RESET_ALL)
print("\n")
print(Fore.BLACK+Back.CYAN+"4. CELEBRITY RECOGNITION"+Style.RESET_ALL)
print("\n")
print(Fore.BLACK+Back.CYAN+"5. MULTIPLE CELEBRITY RECOGNITION"+Style.RESET_ALL)
print("\n")
print(Fore.BLACK+Back.CYAN+"6. TEXT ANALYSIS"+Style.RESET_ALL)

val = input("\nYOUR INPUT PLEASE\n     ")
print("\n")
if val=='1':
    #Object and scene detection    
    with open('credentials.csv','r') as input:
        next(input)
        reader=csv.reader(input)
        for line in reader:
            access_key_id=line[2]
            secret_access_key= line[3]

    photo='burj-khalifa.jpg'
    #Convert to base 64 encoding
    client=boto3.client('rekognition', region_name='ap-south-1',aws_access_key_id=access_key_id,aws_secret_access_key=secret_access_key)

    with open(photo,'rb') as source_image:
        source_bytes=source_image.read()

    response=client.detect_labels(Image={'Bytes':source_bytes},MaxLabels=2,MinConfidence=95)
    print(Fore.MAGENTA+"We used a image of BURJ-KHALIFA that is to be detected by aws rekognition. Lets see the predictions!")
    print(Style.RESET_ALL)
    print(Fore.BLACK + Back.GREEN+"CHECK YOUR TABS TO SEE THE IMAGE THAT WE HAVE SELECTED AND CLOSE THE TAB TO SEE THE PREDICTIONS"+Style.RESET_ALL)
    image = cv2.imread('burj-khalifa.jpg')
    window_name = 'Rekognition Image'
    imS = cv2.resize(image, (960, 540))
    cv2.imshow(window_name,imS)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("The top 2 predicted labels for our image are: \n\n    ",response['Labels'][0]['Name']," with a confidence of ",response['Labels'][0]['Confidence'],end='\n')
    print("\n    ",response['Labels'][1]['Name'],"with a confidence of ",response['Labels'][1]['Confidence'],end='\n')
    source_image.close() 
    
if val=='2':
    #Image moderation
    with open('credentials.csv','r') as input:
        next(input)
        reader=csv.reader(input)
        for line in reader:
            access_key_id=line[2]
            secret_access_key=line[3]

    photo='smoking.jfif'
    #Convert to base 64 encoding
    client=boto3.client('rekognition', region_name='ap-south-1',aws_access_key_id=access_key_id,aws_secret_access_key=secret_access_key)


    with open(photo,'rb') as source_image:
        source_bytes=source_image.read()

    response=client.detect_moderation_labels(Image={'Bytes':source_bytes})
    print(Fore.MAGENTA+"We have used an image of a Smoking & Alocohol that is to be detected by aws rekognition. Lets see what it detects!")
    print(Style.RESET_ALL)
    print(Fore.BLACK + Back.GREEN+"CHECK YOUR TABS TO SEE THE IMAGE THAT WE HAVE SELECTED AND CLOSE THE TAB TO SEE THE PREDICTIONS"+Style.RESET_ALL)
    image = cv2.imread('smoking.jfif')
    window_name = 'Rekognition Image'
    imS = cv2.resize(image, (960, 540))
    cv2.imshow(window_name,imS)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("labels detected are:\n")
    print(response)
    source_image.close()

if val=='3':
    #Facial Analysis
    with open('credentials.csv','r') as input:
        next(input)
        reader=csv.reader(input)
        for line in reader:
            access_key_id=line[2]
            secret_access_key=line[3]

    photo='friends.jfif'
    #Convert to base 64 encoding
    client=boto3.client('rekognition', region_name='ap-south-1',aws_access_key_id=access_key_id,aws_secret_access_key=secret_access_key)


    with open(photo,'rb') as source_image:
        source_bytes=source_image.read()

    response=client.detect_faces(Image={'Bytes':source_bytes},Attributes=['ALL'])
    print(Fore.MAGENTA+"We have used a photo of friends that has 3 members. Lets see the aws rekognition predictions!")
    print(Style.RESET_ALL)
    print(Fore.BLACK + Back.GREEN+"CHECK YOUR TABS TO SEE THE IMAGE THAT WE HAVE SELECTED AND CLOSE THE TAB TO SEE THE PREDICTIONS"+Style.RESET_ALL)
    image = cv2.imread('friends.jfif')
    window_name = 'Rekognition Image'
    imS = cv2.resize(image, (960, 540))
    cv2.imshow(window_name,imS)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("The information found from image analysis are:\n")
    #As multiple faces are detected attributes for each of them are to be displayed
    for key,value in response.items():
        if key=='FaceDetails':
            for people_att in value:
                print(people_att)
                print("================================================================================================================")
    source_image.close()


if val=='4':
    #Celebrity recognition
    with open('credentials.csv','r') as input:
        next(input)
        reader=csv.reader(input)
        for line in reader:
            access_key_id=line[2]
            secret_access_key=line[3]

    photo='celebrity.jfif'
    #Convert to base 64 encoding
    client=boto3.client('rekognition', region_name='ap-south-1',aws_access_key_id=access_key_id,aws_secret_access_key=secret_access_key)


    with open(photo,'rb') as source_image:
        source_bytes=source_image.read()

    response=client.recognize_celebrities(Image={'Bytes':source_bytes})
    print(Fore.MAGENTA+"We have given an input image as the image of  actress Anushka Sharma. Lets see what our aws rekognition detects!")
    print(Style.RESET_ALL)
    print(Fore.BLACK + Back.GREEN+"CHECK YOUR TABS TO SEE THE IMAGE THAT WE HAVE SELECTED AND CLOSE THE TAB TO SEE THE PREDICTIONS"+Style.RESET_ALL)    
    image = cv2.imread('celebrity.jfif')
    window_name = 'Rekognition Image'
    imS = cv2.resize(image, (960, 540))
    cv2.imshow(window_name,imS)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("The celebrity information is as follows:\n")
    for key,value in response.items():
        if key=='CelebrityFaces':
            for people in value:
                print(people)
                print("================================================================================================================")
    source_image.close()

if val=='5':
    #multiple celebrity recognition
    with open('credentials.csv','r') as input:
        next(input)
        reader=csv.reader(input)
        for line in reader:
            access_key_id=line[2]
            secret_access_key=line[3]

    photo_ip='celebrity.jfif'
    photo_op='anushka-virat.jpg'
    #Convert to base 64 encoding
    client=boto3.client('rekognition', region_name='ap-south-1',aws_access_key_id=access_key_id,aws_secret_access_key=secret_access_key)


    with open(photo_ip,'rb') as source_image:
        source_bytes=source_image.read()
    with open(photo_op,'rb') as target_image:
        target_bytes=target_image.read()

    response=client.compare_faces(SourceImage={'Bytes':source_bytes},TargetImage={'Bytes':target_bytes})
    print(Fore.MAGENTA+"We have given an input image of celebrities photo having 5 stars and 3 kids. Anushka Sharma,Virat Kohli and 3 other stars.Then we have tried to match the face of actress Anushka Sharma. Lets see what aws detects!")
    print(Style.RESET_ALL)
    print(Fore.BLACK + Back.GREEN+"CHECK YOUR TABS TO SEE THE IMAGE THAT WE HAVE SELECTED AND CLOSE THE TAB TO SEE THE PREDICTIONS"+Style.RESET_ALL)
    image = cv2.imread('celebrity.jfif')
    image1 = cv2.imread('anushka-virat.jpg')
    window_name = 'Rekognition Image'
    imS = cv2.resize(image, (960, 540))
    imS1 = cv2.resize(image1, (960, 540))
    Hori = np.concatenate((imS, imS1), axis=1)
    imS2 = cv2.resize(Hori, (960, 540))
    cv2.imshow(window_name,imS2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("The matched and unmatched face details are:\n")
    for key,value in response.items():
        if key in ('FaceMatches','UnmatchedFaces'):
            print(key)
            for att in value:
                print(att)
            print("================================================================================================================")
    source_image.close()
    target_image.close()

if val=='6':
    #Text analysis
    with open('credentials.csv','r') as input:
        next(input)
        reader=csv.reader(input)
        for line in reader:
            access_key_id=line[2]
            secret_access_key=line[3]
    photo='qoute.jfif'
    #Convert to base 64 encoding
    client=boto3.client('rekognition', region_name='ap-south-1',aws_access_key_id=access_key_id,aws_secret_access_key=secret_access_key)


    with open(photo,'rb') as source_image:
        source_bytes=source_image.read()

    response=client.detect_text(Image={'Bytes':source_bytes})
    print(Fore.MAGENTA+"We have given a quote for aws to detect. 'It is THE SECRET OF GETTING AHEAD IS GETTING STARTED' Lets see if aws detects this!")
    print(Style.RESET_ALL)
    print(Fore.BLACK + Back.GREEN+"CHECK YOUR TABS TO SEE THE IMAGE THAT WE HAVE SELECTED AND CLOSE THE TAB TO SEE THE PREDICTIONS"+Style.RESET_ALL) 
    image = cv2.imread('qoute.jfif')
    window_name = 'Rekognition Image'
    imS = cv2.resize(image, (960, 540))
    cv2.imshow(window_name,imS)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Texts detected in the image are:\n")
    print(response)
    source_image.close()

print(Fore.GREEN+"\n\nTHANK YOU FOR USING THE AWS REKOGNITION APPLICATION. HAVE A GOOD DAY!!! RUN IT AGAIN TO USE OTHER FEATURES"+Style.RESET_ALL)
print('\n\n')
