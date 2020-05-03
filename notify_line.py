import requests
import os
import sys


def main(calc_time):
    if os.path.exists('line_token.txt'):
        file_token = open('line_token.txt')
        line_notify_token = file_token.read()
        file_token.close()

        line_notify_api = "https://notify-api.line.me/api/notify"
        message = '\nAnimal Classifier Program Finished.【' + \
            str(calc_time) + '】'

        payload = {'message': message}
        headers = {'Authorization': 'Bearer ' + line_notify_token}
        line_notify = requests.post(
            line_notify_api, data=payload, headers=headers)
    else:
        print("Please Setup Line Notify Token.")


if __name__ == "__main__":
    main(0)
