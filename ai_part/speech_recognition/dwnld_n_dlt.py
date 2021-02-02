import firebase_admin
from firebase_admin import credentials, firestore
import base64

cred = credentials.Certificate('online-proctoring-firebase-adminsdk.json')

firebase_admin.initialize_app(cred)

db = firestore.client()

ref = db.collection('snapshots').document(u'audio')


doc = ref.get()
# print(docs)

# for doc in docs:
    # print('{} => {}'.format(doc.id, doc.to_dict()))
data = doc.to_dict()
for k,v in data.items():
    if v[:23] == "data:image/jpeg;base64,":
        with open("assets/" + k + ".jpeg", "wb") as fh:
            fh.write(base64.b64decode(v[23:]))
    elif v[:23] == "data:audio/mp3;;base64,":
        with open("assets/" + k + ".wav", "wb") as fh:
            fh.write(base64.b64decode(v[23:]))
    ref.update({
        k: firestore.DELETE_FIELD
    })


# db.collection(u'snapshots').document(u'nizam').delete()

# doc_ref = db.collection(u'snapshots').document(u'nizam')

# doc = doc_ref.get()
# if doc.exists:
#     print(f'Document data: {doc.to_dict()}')
# else:
#     print(u'No such document!')