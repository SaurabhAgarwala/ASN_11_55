from firebase import Firebase

config = {
  "apiKey": "AIzaSyBzzsBQLEskwb4ly6egRZgEctsjx90b1ig",
  "authDomain": "online-proctoring.firebaseapp.com",
  "databaseURL": "https://online-proctoring.firebaseio.com",
  "storageBucket": "online-proctoring.appspot.com"
}

firebase = Firebase(config)

storage = firebase.storage()
storage.download('MyPic.jpg')

print("Hello World")