import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np


class InfoGenerator:
    def add_text(self, img, text, position, text_color=(0, 255, 0), text_size=30):
        """
        Generate text to display on the image.

        @param img: base map image
        @param text: some messages add on image
        @param position: the position of the upper left corner of the text
        @param text_color: text color
        @param text_size: text size
        @return: a image array include base image and added text
        """
        if isinstance(img, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        canvas = ImageDraw.Draw(img)
        # Change the font style by font style file
        font_style = ImageFont.truetype("./fonts/unic.ttc", text_size, encoding="utf-8")
        canvas.text(position, text, text_color, font=font_style)
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    def draw_ring(self, frame, point_x, point_y, arc_radius=150, start=0, end=360, color=(255, 0, 255), width=20):
        """
        Generate a ring
        @param frame: a video frame or a image.

        @param point_x: the x-coordinate of the center of the ring
        @param point_y: the y-coordinate of the center of the ring
        @param arc_radius: ring radius
        @param start: ring begin angle
        @param end: ring termination angle
        @param color: ring color
        @param width: ring width
        @return: an image array with generated ring
        """
        img = Image.fromarray(frame)
        shape = [(point_x - arc_radius, point_y - arc_radius),
                 (point_x + arc_radius, point_y + arc_radius)]
        img1 = ImageDraw.Draw(img)
        img1.arc(shape, start=start, end=end, fill=color, width=width)
        frame = np.asarray(img)
        return frame

    def generate_ocr_text_area(self, ocr_text, line_text_num, line_num, x, y, w, h, frame):
        """
        Generate OCR text display area in the upper right corner.

        @param ocr_text: the ocr text need to display
        @param line_text_num: the number of characters displayed in a line
        @param line_num: the text line numbers
        @param x: the x-coordinate of the ocr text area
        @param y: the y-coordinate of the ocr text area
        @param w: the width of the ocr text area
        @param h: the height of the ocr text area
        @param frame: a input image array
        @return: the ocr text area with the ocr text
        """
        # First we crop the sub-rect from the image
        sub_img = frame[y: y + h, x: x + w]
        black_rect = np.ones(sub_img.shape, dtype=np.uint8) * 0

        res = cv2.addWeighted(sub_img, 0.5, black_rect, 0.5, 1.0)
        for i in range(line_num):
            text = ocr_text[(i * line_text_num): (i + 1) * line_text_num]
            res = cv2.putText(res, text, (10, 30 * i + 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 0, 0), thickness=2)
        return res

    def generate_label_area(self, text, x, y, w, h, frame):
        """
        Generates a label text image area that identifies the category.

        @param text: the input label text
        @param x: the x-coordinate of the label text area
        @param y: the y-coordinate of the label text area
        @param w: the width of the label text area
        @param h: the height of the label text area
        @param frame: a input image array
        @return: the label text area with the label text include Chinese and English
        """
        # First we crop the sub-rect from the image
        sub_img = frame[y: y + h, x: x + w]
        black_rect = np.ones(sub_img.shape, dtype=np.uint8) * 0

        res = cv2.addWeighted(sub_img, 0.5, black_rect, 0.5, 1.0)
        res = cv2.putText(res, text, (10, 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 0, 0), thickness=2)
        return res




