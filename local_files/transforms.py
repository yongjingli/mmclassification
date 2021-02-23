#add CenterCropPadding and ResizeLongSize mmcls/datasets/pipelines/transforms.py for long size resize

@PIPELINES.register_module()
class CenterCropPadding(object):
    """Center crop the image.

    Args:
        crop_size (int | tuple): Expected size after cropping, (h, w).

    Notes:
        If the image is smaller than the crop size, return the original image
    """

    def __init__(self, crop_size):
        assert isinstance(crop_size, int) or (isinstance(crop_size, tuple)
                                              and len(crop_size) == 2)
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size

    def __call__(self, results):
        crop_height, crop_width = self.crop_size[0], self.crop_size[1]
        for key in results.get('img_fields', ['img']):
            img = results[key]

            # padding short size
            long_size = max(img.shape[:2])
            assert long_size >= max(crop_height, crop_width)
            pad_h = (long_size - img.shape[0]) // 2
            pad_w = (long_size - img.shape[1]) // 2

            img_input = np.ones((long_size, long_size, 3), dtype=np.float32) * 0
            img_input[pad_h:img.shape[0] + pad_h, pad_w:img.shape[1] + pad_w, :] = img
            img = img_input

            # img.shape has length 2 for grayscale, length 3 for color
            img_height, img_width = img.shape[:2]
            y1 = max(0, int(round((img_height - crop_height) / 2.)))
            x1 = max(0, int(round((img_width - crop_width) / 2.)))
            y2 = min(img_height, y1 + crop_height) - 1
            x2 = min(img_width, x1 + crop_width) - 1

            # crop the image
            img = mmcv.imcrop(img, bboxes=np.array([x1, y1, x2, y2]))
            img_shape = img.shape
            results[key] = img
        results['img_shape'] = img_shape

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'



@PIPELINES.register_module()
class ResizeLongSize(object):
    def __init__(self, size, interpolation='bilinear', backend='cv2'):
        assert isinstance(size, int) or (isinstance(size, tuple)
                                         and len(size) == 2)
        self.resize_w_long_side = False
        if isinstance(size, int):
            assert size > 0
            size = (size, size)
        else:
            assert size[0] > 0 and (size[1] > 0 or size[1] == -1)
            if size[1] == -1:
                self.resize_w_long_side = True
        assert interpolation in ('nearest', 'bilinear', 'bicubic', 'area',
                                 'lanczos')
        if backend not in ['cv2', 'pillow']:
            raise ValueError(f'backend: {backend} is not supported for resize.'
                             'Supported backends are "cv2", "pillow"')

        self.size = size
        self.interpolation = interpolation
        self.backend = backend

    def _resize_img(self, results):
        for key in results.get('img_fields', ['img']):
            img = results[key]
            ignore_resize = False
            if self.resize_w_long_side:
                h, w = img.shape[:2]
                long_side = self.size[0]
                if (w >= h and w == long_side) or (h >= w
                                                    and h == long_side):
                    ignore_resize = True
                else:
                    if w > h:
                        width = long_side
                        height = int(long_side * h / w)
                    else:
                        height = long_side
                        width = int(long_side * w / h)
            else:
                height, width = self.size
            if not ignore_resize:
                img = mmcv.imresize(
                    img,
                    size=(width, height),
                    interpolation=self.interpolation,
                    return_scale=False,
                    backend=self.backend)
                results[key] = img
                results['img_shape'] = img.shape
                # print('img_shape:', img.shape)

    def __call__(self, results):
        self._resize_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'interpolation={self.interpolation})'
        return repr_str 
