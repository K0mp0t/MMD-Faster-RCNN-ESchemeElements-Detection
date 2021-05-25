from mmdet.datasets import PIPELINES


@PIPELINES.register_module()
class LevelGrayscaleImage(object):
    """Level grayscale image only.

    This transform makes grayscale image pixels, which are close to white (or black), full white
    (or black). Each pixel are encoded with 1 byte of data ([0...255]) so there could be pixels
    that is close to white (e.g. 253) but not completely white. Same situation with black pixels.
    Those 'close' pixels could be a result of image scanning or any other thing. In either case,
    human eye cannot distinguish shades of gray (e.g. 253 and 255), but for NN the difference
    between those are noticeable, so we are going to increase the number of full black/white
    pixels to make scheme elements borders more distinguishable.

    Args:
        leveling_gap: integer, which defines how "close" pixel should be to 0 (black) or 255
        (white) to be transformed to full black/white (0/255).
    """

    def __init__(self, leveling_gap=20):
        self.leveling_gap = leveling_gap

    def _level_img(self, results):
        for key in results.get('img_fields', ['img']):
            image_array = results[key]
            image_array[image_array < self.leveling_gap] = 0
            image_array[image_array > (255 - self.leveling_gap)] = 255
            results[key] = image_array

    def __call__(self, results):
        self._level_img(results)
        return results
