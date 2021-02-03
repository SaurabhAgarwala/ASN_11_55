import face_recognition

def compare_faces(img1, img2):
    # Load the image
    image1 = face_recognition.load_image_file(img1)
    image2 = face_recognition.load_image_file(img2)

    # Get face encoding
    try:
        image1_encode = face_recognition.face_encodings(image1)[0]
        image2_encode = face_recognition.face_encodings(image2)[0]
        # Compare faces and return True / False
        results = face_recognition.compare_faces([image1_encode], image2_encode)
        print('Encode done....')
        # Return true or false
        return results[0]

    except IndexError as e :
        print(e)


print(compare_faces('assets/haikal.jpg','assets/2525.jpg'))