import sys
sys.path.insert(0, "../")

import tensorrt as trt   # tensorRT 7.0
import pycuda.driver as cuda
import pycuda.autoinit

import numpy as np
import cv2
import time
import os

from mmcls.datasets.builder import DATASETS
image_net = DATASETS.get('ImageNet')
image_net_names = image_net.CLASSES

# You can set the logger severity higher to suppress messages (or lower to display more messages).
# TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)  # set batch size


# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class MmClassificationTrt(object):
    def __init__(self, onnx_path):
        self.onnx_path = onnx_path
        self.engine = self.build_engine_onnx(onnx_path)
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers(self.engine)
        self.context = self.engine.create_execution_context()

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
        print('Mmclassification Trt Init Done.')

    def GiB(self, val):
        return val * 1 << 30

    def build_engine_onnx(self, model_file):
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(
                network, TRT_LOGGER) as parser:
            builder.max_workspace_size = self.GiB(1)
            builder.max_batch_size = 1
            # Load the Onnx model and parse it in order to populate the TensorRT network.
            with open(model_file, 'rb') as model:
                if not parser.parse(model.read()):
                    print('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None
            return builder.build_cuda_engine(network)

    def allocate_buffers(self, engine):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for binding in engine:
            bindig_shape = tuple(engine.get_binding_shape(binding))
            # size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size  # engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(bindig_shape, dtype)
            # print('\tAllocate host buffer: host_mem -> {}, {}'.format(host_mem, host_mem.nbytes))  # host mem

            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # print('\tAllocate device buffer: device_mem -> {}, {}'.format(device_mem, int(device_mem))) # device mem

            # print('\t# Append the device buffer to device bindings.......')
            bindings.append(int(device_mem))
            # print('\tbindings: ', bindings)

            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                # print("this is the input!")
                # print('____HostDeviceMem(host_mem, device_mem)): {}, {}'.format(HostDeviceMem(host_mem, device_mem),type(HostDeviceMem(host_mem, device_mem))))
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                # print("This is the output!")
                outputs.append(HostDeviceMem(host_mem, device_mem))
            # print("----------------------end allocating one binding in the onnx model-------------------------")

        return inputs, outputs, bindings, stream

    def do_inference(self, context, bindings, inputs, outputs, stream, batch_size=1):
        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        # Run inference.
        context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        # Synchronize the stream
        stream.synchronize()
        # Return only the host outputs.
        return [out.host for out in outputs]

    def crop_img_long_size(self, cv_img):
        long_size = max(cv_img.shape[:2])

        pad_h = (long_size - cv_img.shape[0]) // 2
        pad_w = (long_size - cv_img.shape[1]) // 2
        img_input = np.ones((long_size, long_size, 3), dtype=np.uint8) * 0
        img_input[pad_h:cv_img.shape[0] + pad_h, pad_w:cv_img.shape[1] + pad_w, :] = cv_img
        img_input = cv2.resize(img_input, (self.input_size[1], self.input_size[0]), cv2.INTER_LINEAR)

        return img_input

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

    def onnx_input_process(self, cv_img):
        cv_img = self.crop_img_short_size(cv_img)
        # cv_img = self.crop_img_long_size(cv_img)
        assert list(cv_img.shape[:2]) == self.input_size

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
        return self.img_t

    def trt_cv_img_preprocess(self, img_src, pagelocked_buffer):
        # Converts the input image to a CHW Numpy array
        trt_inputs = self.onnx_input_process(img_src)
        np.copyto(pagelocked_buffer, trt_inputs)
        return img_src

    def infer_cv_img(self, cv_img):
        self.trt_cv_img_preprocess(cv_img, self.inputs[0].host)
        trt_outputs = self.do_inference(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream)
        for trt_output in trt_outputs:
            # scale back to origin img
            pred_score = np.max(trt_output, axis=1)[0]
            pred_label = np.argmax(trt_output, axis=1)[0]
            self.result.update({'pred_label': pred_label, 'pred_score': float(pred_score)})

        return self.result


def mm_cls_images_infer(images_dir, onnx_path):
    mm_classifier_trt = MmClassificationTrt(onnx_path)
    image_names = [f for f in os.listdir(images_dir) if os.path.splitext(f)[-1] in ['.jpg', 'png']]
    for image_name in image_names:
        img_path = os.path.join(images_dir, image_name)
        img = cv2.imread(img_path)
        cls_result = mm_classifier_trt.infer_cv_img(img)
        print(cls_result)
        print(image_net.CLASSES[cls_result['pred_label']])
        cv2.namedWindow("img", 0)
        cv2.imshow("img", img)
        cv2.waitKey(0)


if __name__ == "__main__":
    onnx_path = "../checkpoints/imagenet/resnet50_batch256_imagenet_20200708-cfb998bf_sim.onnx"
    images_dir = '../demo'
    mm_cls_images_infer(images_dir, onnx_path)     
