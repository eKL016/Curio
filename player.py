import os

import cv2
import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors
import numpy as np
import torch
from google.cloud import vision
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from PIL import Image

scopes = ["https://www.googleapis.com/auth/youtube.readonly"]
video_path = './hb.mp4'

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./cert.json"
DEVELOPER_KEY = 'AIzaSyDC7V42kQrs6t8Qpsj0s72NRMgnp9RekJM'
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'


def youtube_search(keywords=['Google'], **kwargs):
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION,
                    developerKey=DEVELOPER_KEY)

    search_response = youtube.search().list(
        q=' '.join(keywords),
        part='id,snippet',
        type='video',
        maxResults=kwargs.get('max_results', 5)
    ).execute()

    videos = []

    for search_result in search_response.get('items', []):
        videos.append(
            f'{search_result["snippet"]["title"]} (https://www.youtube.com/watch/?v={search_result["id"]["videoId"]})'
        )
    print('Probably-related Videos:\n', '\n'.join(videos), '\n')


def detect_web(cropped_img_bytestring):
    client = vision.ImageAnnotatorClient()

    image = vision.Image(content=cropped_img_bytestring)

    response = client.web_detection(image=image)
    annotations = response.web_detection

    if annotations.best_guess_labels:
        for label in annotations.best_guess_labels:
            print('\nBest guess label: {}'.format(label.label))

    if annotations.pages_with_matching_images:
        print('\n{} Pages with matching images found:'.format(
            len(annotations.pages_with_matching_images)))

        for page in annotations.pages_with_matching_images:
            print('\n\tPage url   : {}'.format(page.url))

            if page.full_matching_images:
                print('\t{} Full Matches found: '.format(
                    len(page.full_matching_images)))

                for image in page.full_matching_images:
                    print('\t\tImage url  : {}'.format(image.url))

            if page.partial_matching_images:
                print('\t{} Partial Matches found: '.format(
                    len(page.partial_matching_images)))

                for image in page.partial_matching_images:
                    print('\t\tImage url  : {}'.format(image.url))

    if annotations.web_entities:
        labels = []
        print('\n{} Web entities found: '.format(
            len(annotations.web_entities)))
        
        high_acc_cnt = 0
        for entity in annotations.web_entities:
            labels.append(entity.description)
            print('\n\tScore      : {}'.format(entity.score))
            print(u'\tDescription: {}'.format(entity.description))
            if (entity.score>0.8):
              high_acc_cnt = high_acc_cnt + 1
        print('\n')
        if high_acc_cnt > 0:
          youtube_search(labels[:high_acc_cnt+1], max_results=10)
        else:
          youtube_search(labels[:min(len(labels), 10)], max_results=10)
    # if annotations.visually_similar_images:
    #   print('\n{} visually similar images found:\n'.format(
    #     len(annotations.visually_similar_images)))

    #   for image in annotations.visually_similar_images:
    #     print('\tImage url    : {}'.format(image.url))

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))


def draw_detection_results(frame, results_df):
    for res in results_df.itertuples():
        frame = cv2.rectangle(frame,
                              (res.xmin, res.ymin),
                              (res.xmax, res.ymax),
                              (0, 0, 255), 4
                              )
        frame = cv2.putText(frame, res.name,
                            (res.xmin, res.ymin - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 1, cv2.LINE_AA
                            )
        cropped_bytestring = cv2.imencode('.jpg', frame[res.ymin:res.ymax,
                                                        res.xmin:res.xmax])[1].tobytes()
        detect_web(cropped_bytestring)
    return frame


#device = torch.device("mps")
client = vision.ImageAnnotatorClient()
model = torch.hub.load('ultralytics/yolov5', 'yolov5m')
model.conf = 0.4
model.iou = 0.45
cap = cv2.VideoCapture(video_path)


# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video file")

# Read until video is completed
while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        cv2.imshow('Player', frame)
        key_input = cv2.waitKey(25) & 0xFF
        if key_input == ord('q'):
            break
        elif key_input == ord('p'):
            results = model(frame[:, :, ::-1], size=1280)
            results.print()
            results_pandas = results.pandas().xyxy[0]
            results_pandas['xmin'] = results_pandas['xmin'].astype(int)
            results_pandas['ymin'] = results_pandas['ymin'].astype(int)
            results_pandas['xmax'] = results_pandas['xmax'].astype(int)
            results_pandas['ymax'] = results_pandas['ymax'].astype(int)

            print(results_pandas)
            frame_with_res = draw_detection_results(frame, results_pandas)
            cv2.imshow('Player', frame_with_res)
            cv2.waitKey(0)

# Break the loop
    else:
        break

# When everything done, release
# the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
