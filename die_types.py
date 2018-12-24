import imgaug
import numpy as np

XWING_GREEN_DIE_HSV_RANGE = (( 60,  30,  60), ( 90, 255, 255))
XWING_RED_DIE_HSV_RANGE   = ((150,  80,  60), (180, 255, 255))

BLUE_CASINO_DIE_HSV_RANGE = (( 90, 130,  35), (120, 255, 255))
WHITE_DOTS_HSV_RANGE      = ((  0,   0, 200), (255,  30, 255))

class XwingImgTransform:
	def __init__(self):
		self.aug = imgaug.augmenters.Sequential([
			imgaug.augmenters.Affine(
				scale = (0.8, 1.1),
				translate_percent = {"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
				rotate = (0, 360),
				order = 1,
				cval = (0, 255),
			),
			imgaug.augmenters.Sometimes(0.25, imgaug.augmenters.GaussianBlur(sigma=[1.0, 1.8])),
			imgaug.augmenters.AddToHueAndSaturation((-10, 10)),
			#imgaug.augmenters.AdditiveGaussianNoise(loc = 0, scale = (0.0, 0.05*255), per_channel = 0.5)
		])

	def __call__(self, img):
		img = np.array(img)
		return self.aug.augment_image(img)
		
class CasinoImgTransform:
	def __init__(self):
		self.aug = imgaug.augmenters.Sequential([
			imgaug.augmenters.Sometimes(0.75, imgaug.augmenters.Affine(
				scale = (0.8, 1.2),
				translate_percent = {"x": (-0.25, 0.25), "y": (-0.25, 0.25)},
				rotate = (0, 360),
				order = 1,
				cval = (0, 255),
			)),
			imgaug.augmenters.Sometimes(0.25, imgaug.augmenters.GaussianBlur(sigma=[1.0, 1.8])),
			imgaug.augmenters.AddToHueAndSaturation((-5, 5)),
			imgaug.augmenters.AdditiveGaussianNoise(loc = 0, scale = (0.0, 0.02*255), per_channel = 0.5)
		])

	def __call__(self, img):
		img = np.array(img)
		return self.aug.augment_image(img)

# NOTE: Could use namedtuples for each of the elements here, but good enough for now
params = {
	"xwing_red": {
		"hsv_ranges": [XWING_RED_DIE_HSV_RANGE],
		"rect_size": 84,
		"classes_count": 4,			# blank, focus, hit, crit
		"expected_distribution": {"blank": 2.0/8.0, "focus": 2.0/8.0, "hit":   3.0/8.0, "crit":  1.0/8.0},
		"train_image_transform": XwingImgTransform(),
	},
	"xwing_green": {
		"hsv_ranges": [XWING_GREEN_DIE_HSV_RANGE],
		"rect_size": 84,
		"classes_count": 3,			# blank, focus, evade
		"expected_distribution": {"blank": 3.0/8.0, "focus": 2.0/8.0, "evade": 3.0/8.0},
		"train_image_transform": XwingImgTransform(),
	},
	"casino_blue": {
		"hsv_ranges": [BLUE_CASINO_DIE_HSV_RANGE, WHITE_DOTS_HSV_RANGE],
		"rect_size": 100,
		"classes_count": 6,			# 1-6
		"train_image_transform": CasinoImgTransform(),
		"expected_distribution": {"one": 1.0/6.0, "two": 1.0/6.0, "three": 1.0/6.0, "four": 1.0/6.0, "five": 1.0/6.0, "six": 1.0/6.0},
	},
}
