import imgaug
import numpy as np

XWING_GREEN_DIE_HSV_RANGE = (( 60,  30,  60), ( 90, 255, 255))

XWING_RED_DIE_HSV_RANGE_1 = ((  0, 140,  60), ( 10, 255, 255))
XWING_RED_DIE_HSV_RANGE_2 = ((150,  80,  60), (180, 255, 255))

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
			imgaug.augmenters.AddToHueAndSaturation((-20, 20)),
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

class AgeOfWarImgTransform:
	def __init__(self):
		self.aug = imgaug.augmenters.Sequential([
			# We can't do a ton of geometry manipulation given we're using the whole compartment image currently
			imgaug.augmenters.Sometimes(0.5, imgaug.augmenters.Affine(
				scale = (0.9, 1.0),
				translate_percent = {"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
				rotate = (-5, 5),
				order = 1,
				cval = (0, 255),
			)),
			#imgaug.augmenters.Fliplr(0.5),
			#imgaug.augmenters.Flipud(0.5),
			#imgaug.augmenters.Sometimes(0.25, imgaug.augmenters.GaussianBlur(sigma=[1.0, 1.8])),
			imgaug.augmenters.AddToHueAndSaturation((-20, 20)),
		])

	def __call__(self, img):
		img = np.array(img)
		return self.aug.augment_image(img)
		
# NOTE: Could use namedtuples for each of the elements here, but good enough for now
params = {
	"xwing_red": {
		"hsv_ranges": [XWING_RED_DIE_HSV_RANGE_1, XWING_RED_DIE_HSV_RANGE_2],
		"rect_width": 84,
		"rect_height": 84,
		"classes_count": 4,			# blank, focus, hit, crit
		"expected_distribution": {"blank": 2.0/8.0, "focus": 2.0/8.0, "hit":   3.0/8.0, "crit":  1.0/8.0},
		"training": {
			"image_transform": XwingImgTransform(),
			"lr": 0.01,
			"momentum": 0.9,
			"lr_reduction_steps": 30,
			"total_steps": 60,
		},
	},
	"xwing_green": {
		"hsv_ranges": [XWING_GREEN_DIE_HSV_RANGE],
		"rect_width": 84,
		"rect_height": 84,
		"classes_count": 3,			# blank, focus, evade
		"expected_distribution": {"blank": 3.0/8.0, "focus": 2.0/8.0, "evade": 3.0/8.0},
		"training": {
			"image_transform": XwingImgTransform(),
			"lr": 0.01,
			"momentum": 0.9,
			"lr_reduction_steps": 30,
			"total_steps": 60,
		},
	},
	"casino_blue": {
		"hsv_ranges": [BLUE_CASINO_DIE_HSV_RANGE],
		"rect_width": 100,
		"rect_height": 100,
		"classes_count": 6,			# 1-6
		"expected_distribution": {"one": 1.0/6.0, "two": 1.0/6.0, "three": 1.0/6.0, "four": 1.0/6.0, "five": 1.0/6.0, "six": 1.0/6.0},
		"training": {
			"image_transform": CasinoImgTransform(),
			"lr": 0.01,
			"momentum": 0.9,
			"lr_reduction_steps": 30,
			"total_steps": 60,
		},
	},
	"age_of_war": {
		# We don't try and crop the age of war dice since their colors are hard to separate from the background.
		# Instead we'll let the machine learning handle finding the die and identifying it in the image. Less efficient, but simpler for now.
		"hsv_ranges": [((0, 0, 0), (255, 255, 255))],
		"rect_width": 155,
		"rect_height": 388,
		"classes_count": 6,
		"expected_distribution": {"1sword": 1.0/6.0, "2sword": 1.0/6.0, "3sword": 1.0/6.0, "bow": 1.0/6.0, "horse": 1.0/6.0, "mask": 1.0/6.0},
		"training": {
			"image_transform": AgeOfWarImgTransform(),
			"lr": 0.005,
			"momentum": 0.9,
			"lr_reduction_steps": 60,
			"total_steps": 120,
		},
	}
}
