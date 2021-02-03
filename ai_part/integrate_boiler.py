import os, shutil, requests, time

while(True):
    files = os.listdir('assets')
    for file in files:
        candidate_id = file.split('_')[1]
        cheating = False
        file_path = os.path.join('assets',file)
        # process it
        # return result
        url = 'http://localhost:8000/proctor/reportviolation/'+candidate_id+'/violation/'
        print(url)
        requests.get(url)
        file_destination_path = os.path.join("processed", file)
        os.rename(file_path, file_destination_path)
    time.sleep(15)