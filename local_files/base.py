# add self.export in ./mmcls/models/classifers/base.py, modify for onnx export

# 1.self.export
class BaseClassifier(nn.Module, metaclass=ABCMeta):
    """Base class for classifiers"""

    def __init__(self):
        super(BaseClassifier, self).__init__()
        self.export = False

# 2. if export onnx return
if len(imgs) == 1:
    return self.simple_test(imgs[0], **kwargs)
else:
    raise NotImplementedError('aug_test has not been implemented')


def forward(self, img, return_loss=True, **kwargs):
    if self.export:
        return self.forward_test(img, **kwargs)

    """
