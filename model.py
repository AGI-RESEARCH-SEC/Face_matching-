import cv2
import face_recognition
import os

#here we provide the training image for recognition and we have a name of messi.jpg here.
imgMessi = face_recognition.load_image_file('Resources/messi.jpg')

#here we convert our image into RGB format as CV2 deals with RGB color only.
imgMessi = cv2.cvtColor(imgMessi, cv2.COLOR_BGR2RGB)

#Here we encode the face of messi with the function of face_encoding or the face landmark for our train image is stored in encodeMessi.
encodeMessi = face_recognition.face_encodings(imgMessi)[0]

#here we provide the file path of our image.
file_path = 'D:\\Facedetection_agi\\messi_image'

#Here we provide the testing images these images must be stored inside our file path.
image_files = ['messi1.jpg', 'messi2.jpg', 'messi3.jpg']  # Add more image filenames as needed

#using loop for all the image present in our image_files
for file in image_files:
    #Storing the path at which our images is located. os.path.join will concatinate file_path and image_files.
    file_full_path = os.path.join(file_path, file)

    #Loading each file from the file and converting the testing data into RGB again.
    img = face_recognition.load_image_file(file_full_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)



    #face_location function will locate the location of the face.
    face_locations = face_recognition.face_locations(img)
    #face_encoding will encode the landmark of all the images at the required location from the face_location
    face_encodings = face_recognition.face_encodings(img, face_locations)

    #using loop for two funcation namely face_encoding and face_location.
    for face_encoding, face_location in zip(face_encodings, face_locations):
        #Compairing our traning and testing image. It is done by the function compare_faces.
        results = face_recognition.compare_faces([encodeMessi], face_encoding)
        #Locating the distance at which the faces of training and testing image is located.
        face_distance = face_recognition.face_distance([encodeMessi], face_encoding)

        if results[0]:
            #it is the case where the training and testing image match.
            print(f"Messi's picture found in {file} with distance: {round(face_distance[0], 2)}")

            #drawing the rectangle boundry box in the face that is matched and adding a text there.
            cv2.rectangle(img, (face_location[3], face_location[0]), (face_location[1], face_location[2]), (155, 0, 255), 2)
            cv2.putText(img, "Messi", (face_location[3], face_location[2] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (155, 0, 255), 2)
        else:
            #It is the case where training and testing image doesnt match.
            print(f"Messi's picture not found in {file}")

    # Display the image
    cv2.imshow('Image', img)
    cv2.waitKey(0)

cv2.destroyAllWindows()