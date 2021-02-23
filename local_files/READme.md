1. add self.export in ./mmcls/models/classifers/base.py, modify for onnx export \
2.# add LinearClsHeadExport in mmcls/models/heads/linear_head.py for export without softmax for gra-cam 在配置文件需要将cls head替换成该 head \
3.mm_cls_onnx_infer.py 进行onnx 推理 \
4.mm_cls_onnx_export.py输出onnx模型和grad模型 \
5.add '../_base_/datasets/mydata.py', 用于输出的加载的设置 \
6.add ./mmcls/dataset/mydataset.py 新建一个加载数据类，同时需要在__init__.py 文件中载入 \
7.add configs/resnet/resnet50_self.py 主要的模型训练入口设置文件 \
8.#add CenterCropPadding and ResizeLongSize mmcls/datasets/pipelines/transforms.py for long size resize 新增数据变换 \
