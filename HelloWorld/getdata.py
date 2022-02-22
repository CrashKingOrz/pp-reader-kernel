import base64
import json
from django.shortcuts import HttpResponse
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
    def __init__(self, device, window_w=480, window_h=640, out_fps=18):
        self.window_w = window_w
        self.window_h = window_h
        self.out_fps = out_fps
        self.pp_reader = GetHandsInfo(device, window_w, window_h)
        self.image = None
        self.line_len = 100

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
                                                                        self.image, frame_copy)
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
        self.image = self.pp_reader.mode_processor.generator.add_text(self.image, "帧率: " + str(int(fps_text)),
                                                                      (10, 30), text_color=(0, 255, 0), text_size=50)
        self.image = self.pp_reader.mode_processor.generator.add_text(self.image, "手掌: " +
                                                                      str(self.pp_reader.mode_processor.hand_num),
                                                                      (10, 90), text_color=(0, 255, 0), text_size=50)
        self.image = self.pp_reader.mode_processor.generator.add_text(self.image, "模式: " +
                                                                      str(self.pp_reader.mode_processor.hand_mode),
                                                                      (10, 150), text_color=(0, 255, 0), text_size=50)

        return self.image
        # cv2.namedWindow('PPReader', cv2.WINDOW_FREERATIO)
        # cv2.imshow('PPReader', self.image)

        # read
        # self.pp_reader.mode_processor.reader()

        # if cv2.waitKey(5) & 0xFF == 27:
        #     break
        # self.video_cap.release()

ppreader = PPReaderDemo("CPU")

def test(request):

    
    src = json.loads(request.body).get('base64')
    data = src
    image_data = base64.b64decode(data)

    img_np_arr = np.fromstring(image_data, np.uint8)
    print(img_np_arr.shape)
    img_np_arr.resize(640, 480, 4)
    print(img_np_arr.shape)

    ImgRGBA = cv2.cvtColor(img_np_arr,cv2.COLOR_BGR2RGBA)

    # 4 channel -> 3 channel

    image = cv2.cvtColor(ImgRGBA, cv2.COLOR_RGBA2RGB)
    image = ppreader.generate_pp_reader(image)


    cv2.imshow('RandomColor', image)
    cv2.waitKey(10)

    json_data = {'data':'ok'}
    return HttpResponse(json.dumps(json_data, ensure_ascii=False))


