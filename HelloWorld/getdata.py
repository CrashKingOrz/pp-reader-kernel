import base64
import json
from django.http import HttpResponse
import numpy as np
import cv2
import io
from PIL import Image
import time
import cv2
import numpy as np
import logging

from kernel.interface.pp_reader import GetHandsInfo, get_fps_text
from kernel.media.video_processor import get_video_stream, get_mp4_video_writer, frame_operation


class PPReaderDemo:
    def __init__(self, device, window_w=480, window_h=640, out_fps=30):
        self.window_w = window_w
        self.window_h = window_h
        self.out_fps = out_fps
        self.pp_reader = GetHandsInfo(device, window_w, window_h)
        self.image = None
        self.line_len = 100
        self.change_button = 1

    def frame_processor(self):
        if self.pp_reader.results is None:
            logging.info("process result is null, please call hands_model_process function firstly!")

        if self.pp_reader.results.multi_hand_landmarks:
            handedness_list = self.pp_reader.check_hands_index(self.pp_reader.results.multi_handedness)
            hand_num = len(handedness_list)

            self.pp_reader.mode_processor.hand_num = hand_num

            frame_copy = self.image.copy()
            for hand_index, hand_landmarks in enumerate(self.pp_reader.results.multi_hand_landmarks):
                if hand_index > 1:
                    hand_index = 1

                # Label fingers
                self.pp_reader.draw_hands_mark(self.image, hand_landmarks)

                index_finger_tip_x, index_finger_tip_y = self.pp_reader.get_index_finger_tip_axis(hand_landmarks.landmark)

                thumb_finger_tip_x, thumb_finger_tip_y = self.pp_reader.get_thumb_finger_tip_axis(hand_landmarks.landmark)
                self.image = self.pp_reader.draw_paw_box(self.image, hand_landmarks.landmark, handedness_list, hand_index)
                self.image = self.pp_reader.mode_processor.mode_execute(handedness_list[hand_index],
                                                                        [index_finger_tip_x, index_finger_tip_y],[thumb_finger_tip_x, thumb_finger_tip_y],
                                                                        self.image, frame_copy, self.change_button)
        else:
            self.pp_reader.mode_processor.none_mode()
        return self.image

    def generate_pp_reader(self, image):
        # using time to calculate fps
        fps_time = time.time()

        self.image = image

        self.image = cv2.resize(self.image, (self.window_w, self.window_h))
        # close write mode to improve performance
        self.image.flags.writeable = False
        # rotate or flip or None
        self.image = frame_operation(self.image, rotate=False, flip=False)

        # mediapipe mode process
        self.pp_reader.hands_model_process(self.image)

        self.image.flags.writeable = True
        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)

        # read
        # self.pp_reader.mode_processor.reader()

        # Todo: store thumbnail, this function need to realize this file
        if isinstance(self.pp_reader.mode_processor.last_thumb_img, np.ndarray):
            self.image = self.pp_reader.mode_processor.generate_thumbnail(
                self.pp_reader.mode_processor.last_thumb_img, self.image)

        self.image = self.frame_processor()

        # 显示刷新率FPS
        ctime = time.time()
        fps_text = get_fps_text(ctime, fps_time)
        fps_time = ctime

        self.image = cv2.putText(self.image, "fps: " + str(int(fps_text)), (10, 30),
                                 fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=2)
        self.image = cv2.putText(self.image, "paw: " + str(self.pp_reader.mode_processor.hand_num),
                                 (10, 90), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0),
                                 thickness=2)
        self.image = cv2.putText(self.image, "mode: " + str(self.pp_reader.mode_processor.hand_mode),
                                 (10, 150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0),
                                 thickness=2)

        # cv2.namedWindow('PPReader', cv2.WINDOW_FREERATIO)
        # cv2.imshow('PPReader', self.image)

        # read
        # self.pp_reader.mode_processor.reader()

        return self.image

    def change_mode(self, mode):
        if mode < 2:
            self.change_button = mode
            return True
        else:
            return False


ppreader = PPReaderDemo("GPU")

import time
ctime = time.time()
last_fps = 20


def process_fps(fps):
    global last_fps

    if fps > 30:
        fps = last_fps

    last_fps = fps

    return fps


def test(request):
    global ctime
    
    src = json.loads(request.body).get('base64')
    mode = json.loads(request.body).get('mode')
    # print("mode: ", mode)

    data = src
    image_data = base64.b64decode(data)

    img_np_arr = np.fromstring(image_data, np.uint8)
    img_np_arr = cv2.imdecode(img_np_arr, cv2.IMREAD_COLOR)
    # print(img_np_arr.shape)

    json_data = {'state': 'ERROR: No Image Received!'}

    if img_np_arr is not None:

        change_result = ppreader.change_mode(mode)
        if not change_result:
            print("ERROR: do not support the mode!")

        image = img_np_arr.copy()
        image = ppreader.generate_pp_reader(image)
        # cv2.imwrite("test.jpg", image)

        json_data['state'] = 'working'
        json_data['hands'] = ppreader.pp_reader.mode_processor.get_hands_num()

        if mode == 0:  # 识字：点
            text_result = ppreader.pp_reader.mode_processor.get_text_result()
            index_tip_coordinates = ppreader.pp_reader.mode_processor.get_index_tip_coordinates()

            result_data = {'text': text_result, 'keypoint': index_tip_coordinates}
            json_data.update(result_data)

        elif mode == 1:  # 识物：框
            detection_label = ppreader.pp_reader.mode_processor.get_detection_label()
            # ocr_text = ppreader.pp_reader.mode_processor.get_ocr_text()
            recognize_area = ppreader.pp_reader.mode_processor.get_recognize_area()

            result_data = {'text': detection_label, 'keypoint': recognize_area}
            json_data.update(result_data)

        #cv2.imshow('RandomColor', image)
        #cv2.waitKey(1)

    else:
        print("ERROR: No Image Received!")

    fps = 1.0/(time.time()-ctime)
    fps = process_fps(fps)

    ppreader.pp_reader.mode_processor.change_duration(fps)

    ctime = time.time()
    json_data['fps'] = int(fps)

    print(json_data)

    return HttpResponse(json.dumps(json_data, ensure_ascii=False))


