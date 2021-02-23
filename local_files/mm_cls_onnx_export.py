import sys
sys.path.insert(0, "../")

import numpy as np
import cv2
import argparse
import torch
from torchvision.transforms import functional as F
from mmcls.apis import init_model
import os
from tqdm import tqdm


class MmClassifier(object):
    def __init__(self, cls_c, cls_w, device):
        self.model_w = cls_w
        self.device = device
        self.cls_model = init_model(cls_c, cls_w, device=device)
        self.cls_model.export = True    # set export and return convolution result
        self.short_size = 256
        self.dst_w = 224
        self.dst_h = 224
        self.input_size = [self.dst_h, self.dst_w]

        self.mean = np.array([123.675, 116.28, 103.53])
        self.std = np.array([58.395, 57.12, 57.375])
        self.std_inv = 1/self.std

        self.cls_name = self.cls_model.CLASSES
        self.result = dict()

    def crop_img_short_size(self, cv_img):
        # resize the short size
        h, w, _ = cv_img.shape
        print(h, w)
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
        # cv_img = self.crop_img_short_size(cv_img)
        cv_img = self.crop_img_long_size(cv_img)
        assert list(cv_img.shape[:2]) == self.input_size

        # normalize
        cv_img = cv_img.copy().astype(np.float32)
        self.mean = np.float64(self.mean.reshape(1, -1))
        self.std_inv = 1 / np.float64(self.std.reshape(1, -1))
        if True:
            cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB, cv_img)  # inplace
        cv2.subtract(cv_img, self.mean, cv_img)  # inplace
        cv2.multiply(cv_img, self.std_inv, cv_img)  # inplace

        with torch.no_grad():
            self.img_t = F.to_tensor(cv_img.copy())  # no resize
            self.img_t = self.img_t.unsqueeze(0)

            if 'cpu' not in self.device:
                self.img_t = self.img_t.cuda()

            output = self.cls_model(self.img_t, return_loss=False)  # forward

            pred_score = np.max(output, axis=1)[0]
            pred_label = np.argmax(output, axis=1)[0]
            self.result.update({'pred_label': pred_label, 'pred_score': float(pred_score), 'pred_class':self.cls_name[pred_label]})
        return self.result

    #
    def export_onnx_model(self):
        self.cls_model.export = True

        img_dry = torch.zeros((1, 3, self.dst_h, self.dst_w))
        with torch.no_grad():
            y = self.cls_model(img_dry, return_loss=False)  # forward

        try:
            import onnx
            print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
            f = self.model_w.replace('.pth', '.onnx')  # filename
            # print(model.t)
            torch.onnx.export(self.cls_model, img_dry, f, verbose=False, opset_version=11, \
                              input_names=['images'], output_names=['output'])
            # Checks
            onnx_model = onnx.load(f)  # load onnx model
            onnx.checker.check_model(onnx_model)  # check onnx model
            print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model

            # simpily onnx
            from onnxsim import simplify
            model_simp, check = simplify(onnx_model)
            assert check, "Simplified ONNX model could not be validated"

            f2 = f.replace('.onnx', '_sim.onnx')  # filename
            onnx.save(model_simp, f2)
            print('====ONNX SIM export success, saved as %s' % f2)

            from onnx import shape_inference
            f3 = f2.replace('.onnx', '_shape.onnx')  # filename
            onnx.save(onnx.shape_inference.infer_shapes(onnx.load(f2)), f3)
            print('====ONNX shape inference export success, saved as %s' % f3)

            print('ONNX export success, saved as %s' % f)
        except Exception as e:
            print('ONNX export failure: %s' % e)

    def export_pth_model(self):
        self.cls_model.export = True
        img_dry = torch.zeros((1, 3, self.dst_h, self.dst_w))
        with torch.no_grad():
            y = self.cls_model(img_dry)  # forward

        # torch.save(self.cls_model.state_dict(), "my_model.pth")  # 只保存模型的参数
        f = self.model_w.replace('.pth', '_gram.pth')  # filename
        print("save pth model:", f)
        torch.save(self.cls_model, f)  # 保存整个模型


if __name__ == "__main__":
    print("Start Porc...")
    parser = argparse.ArgumentParser()
    parser.add_argument('--pose_config', help='Config file for detection')
    parser.add_argument('--pose_checkpoint', help='Checkpoint file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    opt = parser.parse_args()

    opt.pose_config = "../configs/resnet/resnet50_b32x8_imagenet.py"
    opt.pose_checkpoint = "../checkpoints/imagenet/resnet50_batch256_imagenet_20200708-cfb998bf.pth"
    opt.device = "cpu"
    mm_classifier = MmClassifier(opt.pose_config, opt.pose_checkpoint, opt.device)

    img_path = "demo/cat.jpg"
    img = cv2.imread(img_path)

    # export onnx
    # mm_classifier.export_onnx_model()

    mm_classifier.export_pth_model()
    exit(1)

    # Start pose detector
    img_path = "/home/liyongjing/Egolee/programs/mmclassification-master/demo/cat.jpg"
    root_dir = "/home/liyongjing/Egolee/data/dataset/cls_fall_down/fall_down_person"
    dst_dir = "/home/liyongjing/Egolee/programs/mmclassification-master/data/cls_fall_down/find_fall_down_from_opendataset/crowd_person_train"
    img_names = list(filter(lambda x: os.path.splitext(x)[-1] in [".jpg"], os.listdir(root_dir)))
    for img_name in tqdm(img_names):
        img_path = os.path.join(root_dir, img_name)
        img = cv2.imread(img_path)
        cls_result = mm_classifier.get_cls_result(img)
        print(cls_result)

        cv2.namedWindow("img", 0)
        cv2.imshow("img", img)
        if cls_result['pred_label'] != 1:
            # dst_path = img_path.replace(root_dir, dst_dir)
            # shutil.copy(img_path, dst_path)
            wait_key = cv2.waitKey(0)
            if wait_key == 27:
                exit(1)


    # pred_result = mm_classifier.get_cls_result(img)
    # print(pred_result)

    # cv2.namedWindow("img", 0)
    # cv2.imshow("img", img)
    # wait_key = cv2.waitKey(0)
    # if wait_key == 0:
    #     exit(1)  
