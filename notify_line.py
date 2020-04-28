import requests
import os
import sys

if os.path.exists('line_token.txt'):
    file_token = open('line_token.txt')
    line_notify_token = file_token.read()
    file_token.close()

    line_notify_api = "https://notify-api.line.me/api/notify"
    if len(sys.argv) == 2:
        message = '\nAnimal Classifier Program Finished. \n 【' + sys.argv[1]  + '】'
    else:
        message = '\nTractography Program Finished.'

    payload = {'message': message}
    headers = {'Authorization': 'Bearer ' + line_notify_token}
    line_notify = requests.post(line_notify_api, data=payload, headers=headers)
else:
    print("Please Setup Line Notify Token.")