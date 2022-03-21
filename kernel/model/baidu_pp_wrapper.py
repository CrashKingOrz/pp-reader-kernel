import numpy as np
import cv2
import jieba

from kernel.model.baidu_pp_ocr.tools.infer import utility
from kernel.model.baidu_pp_detection.python.infer import Config, Detector
from kernel.model.baidu_pp_detection.python.visualize import visualize_box_mask, lmk2out
from kernel.model.baidu_pp_ocr.tools.infer.predict_system import TextSystem
from kernel.model.baidu_pp_ocr.ppocr.utils.logging import get_logger

logger = get_logger()


class PpDetection:
    # Object recognition
    def __init__(self, device="GPU"):
        """
        Initialize detection models.

        """
        self.model_dir = './kernel/model/baidu_pp_detection/models/cascade_rcnn_dcn_r101_vd_fpn_gen_server_side'
        config = Config(self.model_dir)
        self.labels_en = config.labels
        self.labels_zh = self.get_label_zh()
        self.ob_detector = Detector(
            config,
            self.model_dir,
            device=device,
            run_mode='fluid',
            trt_calib_mode=False)

        # Warm up detection model
        if 1:
            print('Warm up detection model')
            img = np.random.uniform(0, 255, [640, 640, 3]).astype(np.uint8)
            for i in range(3):
                im, results = self.detect_img(img)

    def get_label_zh(self):
        """
        Obtain the corresponding Chinese label.

        @return back_list: Chinese label list
        """
        file_path = self.model_dir + '/generic_det_label_list_zh.txt'
        back_list = []
        with open(file_path, 'r', encoding='utf-8') as label_text:
            for label in label_text.readlines():
                back_list.append(label.replace('\n', ''))
        return back_list

    def detect_img(self, img):
        """
        Detect the image.

        @param img: Represents the three-dimensional matrix of the image
        @return im: Represents the three-dimensional matrix of the image with boxes and label
        @return results: the label of img

        """
        results = self.ob_detector.predict(img, 0.5)
        im = visualize_box_mask(
            img,
            results,
            self.ob_detector.config.labels,
            mask_resolution=self.ob_detector.config.mask_resolution,
            threshold=0.5)
        im = np.array(im)
        return im, results


class PpOCR:
    # OCR
    def __init__(self, device='GPU'):
        """
        Initialize ocr model.

        """
        args = utility.parse_args()
        args.det_model_dir = "./kernel/model/baidu_pp_ocr/models/ch_PP-OCRv2_det_infer/"
        args.rec_model_dir = "./kernel/model/baidu_pp_ocr/models/ch_PP-OCRv2_rec_infer/"
        args.rec_char_dict_path = "./kernel/model/baidu_pp_ocr/ppocr/utils/ppocr_keys_v1.txt"
        args.use_angle_cls = False
        self.text_sys = TextSystem(args)

        # gpu or cpu
        if device == "GPU":
            args.use_gpu = True
        elif device == "CPU":
            args.use_gpu = False
        else:
            logger.error("Error: device should be GPU or CPU!")

        # Warm up ocr model
        if 1:
            print('Warm up ocr model')
            img = np.random.uniform(0, 255, [640, 640, 3]).astype(np.uint8)
            for i in range(10):
                res = self.text_sys(img)

    def ocr_image(self, img):
        """
        Read image text.

        @param img: Represents the three-dimensional matrix of the image
        @return src_im:  Represents the three-dimensional matrix of the image with boxes and textlabel
        @return text_list:  the textlabel of img
        """
        dt_boxes, rec_res = self.text_sys(img)
        text_list = []
        for text, score in rec_res:
            text_list.append(text)
        src_im = img

        for box in dt_boxes:
            box = np.array(box).astype(np.int32).reshape(-1, 2)
            cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)

        return src_im, text_list

    def ocr_image_one_word_or_sentence(self, img, index_finger_tip_coordinates=np.array([0, 0]), flag='word'):
        """
        Recognize one word or one sentence.

        @param img: Represents the three-dimensional matrix of the image
        @return src_im:  Represents the three-dimensional matrix of the image with boxes and textlabel
        @return text_list:  the textlabel of img
        """

        def point_line_distance(point, line_point1, line_point2):
            # 计算向量
            vec1 = line_point1 - point
            vec2 = line_point2 - point
            distance = np.abs(np.cross(vec1, vec2)) / np.linalg.norm(line_point1 - line_point2)
            return distance

        def get_single_character_average_length(box, words_num):
            row_length = box[2][0] - box[3][0]
            single_character_average_length = row_length / words_num
            return single_character_average_length

        dt_boxes, rec_res = self.text_sys(img)
        text_list = []
        for text, score in rec_res:
            text_list.append(text)

        left_top_x = left_top_y = right_bottom_x = right_bottom_y = 0

        image = img
        text_result = []

        if flag == 'sentence':
            text_result = text_list

        if len(text_list):
            d_list = []
            text_result = []
            for box in dt_boxes:
                d1 = point_line_distance(index_finger_tip_coordinates, box[2], box[3])
                if box[3][0] <= index_finger_tip_coordinates[0] <= box[2][0] and \
                        (index_finger_tip_coordinates[1] < box[0][1] or index_finger_tip_coordinates[1] < box[1][1]):
                    d2 = 0
                else:
                    d2 = 10000
                d_list.append(d1 + d2)

            d_list = np.array(d_list)
            row_index = np.argmin(d_list)

            if flag == 'character':
                single_character_length = get_single_character_average_length(dt_boxes[row_index], len(text_list[row_index]))
                character_index = int(
                    (index_finger_tip_coordinates[0] - dt_boxes[row_index][3][0]) / single_character_length) - 1
                if character_index < 0:
                    character_index = 0
                if character_index >= len(text_list[row_index]):
                    character_index = len(text_list[row_index]) - 1
                text_result = text_list[row_index][character_index]

                left_top_x = dt_boxes[row_index][0][0] + character_index * single_character_length
                left_top_y = dt_boxes[row_index][0][1]
                right_bottom_x = dt_boxes[row_index][2][0] + (character_index + 1) * single_character_length
                right_bottom_y = dt_boxes[row_index][2][1]

            elif flag == 'word':
                sentence = text_list[row_index]
                words = jieba.lcut(sentence)
                # print(words)

                single_character_length = get_single_character_average_length(dt_boxes[row_index], len(text_list[row_index]))
                total_length = dt_boxes[row_index][3][0]
                for word in words:
                    total_length += len(word)*single_character_length
                    if total_length >= index_finger_tip_coordinates[0]:
                        text_result = word

                        left_top_x = dt_boxes[row_index][0][0] + total_length - single_character_length * len(word)
                        left_top_y = dt_boxes[row_index][0][1]
                        right_bottom_x = dt_boxes[row_index][2][0] + total_length
                        right_bottom_y = dt_boxes[row_index][2][1]

                        break

            elif flag == 'sentence':
                text_result = text_list[row_index]

                left_top_x = dt_boxes[row_index][0][0]
                left_top_y = dt_boxes[row_index][0][1]
                right_bottom_x = dt_boxes[row_index][2][0]
                right_bottom_y = dt_boxes[row_index][2][1]


            left_top_x = int(max(left_top_x - 2, 0))
            left_top_y = int(max(left_top_y - 2, 0))
            right_bottom_x = int(min(right_bottom_x + 2, img.shape[0]))
            right_bottom_y = int(min(right_bottom_y + 2, img.shape[1]))

            image = img[left_top_y:right_bottom_y, left_top_x:right_bottom_x, :]
            image = cv2.rectangle(image, (left_top_x, left_top_y), (right_bottom_x, right_bottom_y),
                                  (0, 255, 0), 2)
            # print(text_result)

        return image, text_result



