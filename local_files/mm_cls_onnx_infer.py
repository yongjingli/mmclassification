import sys
sys.path.insert(0, "../")

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import cv2
import onnxruntime
import numpy as np
import os

from mmcls.datasets.builder import DATASETS
image_net = DATASETS.get('ImageNet')
image_net_names = image_net.CLASSES


class MmClassifierOnnx():
    def __init__(self, onnx_path):
        self.onnx_path = onnx_path
        self.ort_sess = onnxruntime.InferenceSession(self.onnx_path)
        self.input_name = self.ort_sess.get_inputs()[0].name

        self.short_size = 256
        self.dst_w = 224
        self.dst_h = 224
        self.input_size = [self.dst_h, self.dst_w]

        self.long_size = 224

        self.mean = np.array([123.675, 116.28, 103.53])
        self.std = np.array([58.395, 57.12, 57.375])
        self.std_inv = 1 / self.std
        self.img_t = None

        self.result = dict()

    def crop_img_short_size(self, cv_img):
        # resize the short size
        h, w, _ = cv_img.shape
        if h >= w:
            h = int(h * self.short_size / w)
            w = int(self.short_size)
        else:
            w = int(w * self.short_size / h)
            h = int(self.short_size)

        cv_img = cv2.resize(cv_img, (w, h), cv2.INTER_LINEAR)

        # center crop
        y1 = max(0, int(round((h - self.input_size[1]) / 2.)))
        x1 = max(0, int(round((w - self.input_size[0]) / 2.)))
        y2 = min(h-1, y1 + self.input_size[1])
        x2 = min(w-1, x1 + self.input_size[0])

        cv_img = cv_img[y1:y2, x1:x2, :]
        return cv_img

    def crop_img_long_size(self, cv_img):
        long_size = max(cv_img.shape[:2])

        pad_h = (long_size - cv_img.shape[0]) // 2
        pad_w = (long_size - cv_img.shape[1]) // 2
        img_input = np.ones((long_size, long_size, 3), dtype=np.uint8) * 0
        img_input[pad_h:cv_img.shape[0] + pad_h, pad_w:cv_img.shape[1] + pad_w, :] = cv_img
        img_input = cv2.resize(img_input, (self.input_size[1], self.input_size[0]), cv2.INTER_LINEAR)

        return img_input

    def get_cls_result(self, cv_img):
        cv_img = self.crop_img_short_size(cv_img)
        # cv_img = self.crop_img_long_size(cv_img)
        assert list(cv_img.shape[:2]) == self.input_size

        cv2.namedWindow("cv_img", 0)
        cv2.imshow("cv_img", cv_img)

        # normalize
        cv_img = cv_img.copy().astype(np.float32)
        self.mean = np.float64(self.mean.reshape(1, -1))
        self.std_inv = 1 / np.float64(self.std.reshape(1, -1))
        if True:
            cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB, cv_img)  # inplace
        cv2.subtract(cv_img, self.mean, cv_img)  # inplace
        cv2.multiply(cv_img, self.std_inv, cv_img)  # inplace

        self.img_t = cv_img.transpose(2, 0, 1)  # to C, H, W
        self.img_t = np.ascontiguousarray(self.img_t)

        self.img_t = np.expand_dims(self.img_t, axis=0)

        output = self.ort_sess.run(None, {self.input_name: self.img_t})[0]

        pred_score = np.max(output, axis=1)[0]
        pred_label = np.argmax(output, axis=1)[0]
        self.result.update({'pred_label': pred_label, 'pred_score': float(pred_score)})
        return self.result


def mm_cls_images_infer(images_dir, onnx_path):
    mm_classifier_onnx = MmClassifierOnnx(onnx_path)
    image_names = [f for f in os.listdir(images_dir) if os.path.splitext(f)[-1] in ['.jpg', 'png']]
    for image_name in image_names:
        img_path = os.path.join(images_dir, image_name)
        img = cv2.imread(img_path)
        cls_result = mm_classifier_onnx.get_cls_result(img)
        print(cls_result)
        print(image_net.CLASSES[cls_result['pred_label']])
        cv2.namedWindow("img", 0)
        cv2.imshow("img", img)
        cv2.waitKey(0)


if __name__ == "__main__":
    logger.info("Start Proc...")
    onnx_path = "../checkpoints/imagenet/resnet50_batch256_imagenet_20200708-cfb998bf_sim.onnx"
    images_dir = '../demo'
    mm_cls_images_infer(images_dir, onnx_path) 
