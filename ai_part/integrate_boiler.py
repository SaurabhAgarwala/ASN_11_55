import os, shutil, requests, time

while(True):
    files = os.listdir('assets')
    for file in files:
        candidate_id = file.split('_')[1]
        cheating = False
        file_path = os.path.join('demo_assets',file)
        # process it
        # return result
        requests.get('http://localhost:8000/proctor/reportviolation/'+candidate_id+'/violation')
        file_destination_path = os.path.join("processed", file)
        os.rename(file_path, file_destination_path)
    time.sleep(15)