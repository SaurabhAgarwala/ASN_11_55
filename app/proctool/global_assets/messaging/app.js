var config = {
    apiKey: "AIzaSyBzzsBQLEskwb4ly6egRZgEctsjx90b1ig",
    authDomain: "online-proctoring.firebaseapp.com",
    projectId: "online-proctoring",
    storageBucket: "online-proctoring.appspot.com",
    messagingSenderId: "812057776891",
    appId: "1:812057776891:web:66bca88a7867d94b2fd65e",
}

firebase.initializeApp(config);
const messaging = firebase.messaging();
messaging.requestPermission()
    .then(function() {
        console.log('Have permission');
    })
    .catch(function(err){
        console.log(' Error no cloud messaging permission');
    })