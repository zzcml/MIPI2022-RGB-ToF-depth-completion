import numpy as np
from PIL import Image


# Define cv2 interpolation constants if cv2 is not available
try:
    import cv2
    INTER_CUBIC = cv2.INTER_CUBIC
    INTER_AREA = cv2.INTER_AREA
    INTER_LINEAR = cv2.INTER_LINEAR
    INTER_NEAREST = cv2.INTER_NEAREST
except ImportError:
    # Fallback using PIL interpolation
    INTER_CUBIC = 3  # Approximate value for BICUBIC
    INTER_AREA = 3   # Approximate value for AREA
    INTER_LINEAR = 2 # Approximate value for BILINEAR
    INTER_NEAREST = 0  # Approximate value for NEAREST


def cv2_resize(image, size, interpolation=INTER_LINEAR):
    """Resize image using cv2 or PIL as fallback."""
    try:
        import cv2
        return cv2.resize(image, size, interpolation=interpolation)
    except ImportError:
        # Use PIL as fallback
        pil_image = Image.fromarray(image)
        if interpolation == INTER_CUBIC:
            pil_resized = pil_image.resize(size, Image.BICUBIC)
        elif interpolation == INTER_AREA:
            pil_resized = pil_image.resize(size, Image.LANCZOS)
        elif interpolation == INTER_NEAREST:
            pil_resized = pil_image.resize(size, Image.NEAREST)
        else:
            pil_resized = pil_image.resize(size, Image.BILINEAR)
        return np.array(pil_resized)


class Resize(object):
    """Resize sample to given size (width, height).
    """

    def __init__(
        self,
        width,
        height,
        resize_target=True,
        keep_aspect_ratio=False,
        ensure_multiple_of=1,
        resize_method="lower_bound",
        image_interpolation_method=INTER_AREA,
    ):
        """Init.

        Args:
            width (int): desired output width
            height (int): desired output height
            resize_target (bool, optional):
                True: Resize the full sample (image, mask, target).
                False: Resize image only.
                Defaults to True.
            keep_aspect_ratio (bool, optional):
                True: Keep the aspect ratio of the input sample.
                Output sample might not have the given width and height, and
                resize method fits into the given rectangle.
                Defaults to False.
            ensure_multiple_of (int, optional):
                Ensure that the output width and height are multiples of this value.
                Defaults to 1.
            resize_method (str, optional):
                "lower_bound": Scale up to at least the given size.
                "upper_bound": Scale down to at most the given size.
                "minimal": Scale to the minimal size.
                Defaults to "lower_bound".
            image_interpolation_method (cv2 interpolation method, optional):
                Interpolation method for image resizing.
                Defaults to INTER_AREA.
        """
        self.__width = width
        self.__height = height
        self.__resize_target = resize_target
        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__ensure_multiple_of = ensure_multiple_of
        self.__resize_method = resize_method
        self.__image_interpolation_method = image_interpolation_method

    def constrain_to_multiple_of(self, val, min_val=None, max_val=None):
        """Constrain value to be a multiple of a given number."""
        if max_val is not None and val > max_val:
            val = max_val
        if min_val is not None and val < min_val:
            val = min_val
        
        return (val // self.__ensure_multiple_of) * self.__ensure_multiple_of

    def get_size(self, width, height):
        """Get the new size based on the resize method."""
        if self.__keep_aspect_ratio:
            scale_width = self.__width / width
            scale_height = self.__height / height
            
            if self.__resize_method == "lower_bound":
                # scale as little as possible to fit both dimensions
                if scale_width > scale_height:
                    scale_height = scale_width
                else:
                    scale_width = scale_height
            elif self.__resize_method == "upper_bound":
                # scale as much as possible to fit both dimensions
                if scale_width < scale_height:
                    scale_height = scale_width
                else:
                    scale_width = scale_height
            elif self.__resize_method == "minimal":
                # scale as least as possbile
                if abs(1 - scale_width) < abs(1 - scale_height):
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            else:
                raise ValueError(f"resize_method {self.__resize_method} not implemented")

        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(scale_height * height, min_val=self.__height)
            new_width = self.constrain_to_multiple_of(scale_width * width, min_val=self.__width)
        elif self.__resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(scale_height * height, max_val=self.__height)
            new_width = self.constrain_to_multiple_of(scale_width * width, max_val=self.__width)
        elif self.__resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError(f"resize_method {self.__resize_method} not implemented")

        return (new_width, new_height)

    def __call__(self, sample):
        width, height = self.get_size(sample["image"].shape[1], sample["image"].shape[0])

        # resize sample using cv2 or PIL fallback
        sample["image"] = cv2_resize(sample["image"], (width, height), interpolation=self.__image_interpolation_method)

        if self.__resize_target:
            if "depth" in sample:
                sample["depth"] = cv2_resize(sample["depth"], (width, height), interpolation=INTER_NEAREST)

            if "mask" in sample:
                sample["mask"] = cv2_resize(sample["mask"].astype(np.float32), (width, height), interpolation=INTER_NEAREST)

        return sample


class NormalizeImage(object):
    """Normlize image by given mean and std.
    """

    def __init__(self, mean, std):
        self.__mean = np.array(mean, dtype=np.float32)
        self.__std = np.array(std, dtype=np.float32)

    def __call__(self, sample):
        image = sample["image"]
        image = (image - self.__mean) / self.__std
        sample["image"] = image
        return sample


class PrepareForNet(object):
    """Prepare sample for network input.
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        image = sample["image"]
        image = image.transpose(2, 0, 1)
        sample["image"] = image.astype(np.float32)
        return sample
