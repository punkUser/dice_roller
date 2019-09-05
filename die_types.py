import imgaug
import numpy as np

XWING_GREEN_DIE_HSV_RANGE = (( 60,  30,  60), ( 90, 255, 255))

XWING_RED_DIE_HSV_RANGE_1 = ((  0, 140,  60), ( 10, 255, 255))
XWING_RED_DIE_HSV_RANGE_2 = ((150,  80,  60), (180, 255, 255))

XWING_RED_GD_DIE_HSV_RANGE_1 = ((  0,  60,  20), ( 10, 255, 255))
XWING_RED_GD_DIE_HSV_RANGE_2 = ((155,  60,  20), (185, 255, 255))

# WIP, can't really get these working acceptably
XWING_RED_BLACK_DIE_HSV_RANGE_1 = ((150, 20,  0), (180, 255, 255))
XWING_RED_BLACK_DIE_HSV_RANGE_2 = (( 80,  0, 10), (180, 255,  50))

BLUE_CASINO_DIE_HSV_RANGE = (( 90, 130,  35), (120, 255, 255))
WHITE_DOTS_HSV_RANGE      = ((  0,   0, 120), (255,  30, 255))

D8_BLUE_DIE_HSV_RANGE     = ((80, 130,   0), (130, 255, 255))

# Need two ranges since sometimes we get a fairly high specular reflection on these metal dice
D8_ORANGE_DIE_HSV_RANGE_1 = (( 0,  125, 100), ( 20, 255, 255))
D8_ORANGE_DIE_HSV_RANGE_2 = (( 0,   80, 160), ( 20, 255, 255))
		
class XwingImgTransform:
	def __init__(self):
		self.aug = imgaug.augmenters.Sequential([
			imgaug.augmenters.Sometimes(0.2, imgaug.augmenters.CoarseDropout((0.01, 0.05), size_percent=(0.10, 0.25))),
			imgaug.augmenters.Affine(
				scale = (0.8, 1.1),
				translate_percent = {"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
				rotate = (0, 360),
				order = 1,
				cval = (0, 255),
			),
			#imgaug.augmenters.Sometimes(0.25, imgaug.augmenters.GaussianBlur(sigma=[1.0, 1.8])),
			imgaug.augmenters.Sometimes(0.5, imgaug.augmenters.Add((0, 100))),
			imgaug.augmenters.Sometimes(0.5, imgaug.augmenters.Multiply((0.9, 1.5))),
			imgaug.augmenters.Sometimes(0.3, imgaug.augmenters.Grayscale([0.5, 1.0])),
		])

	def __call__(self, img):
		img = np.array(img)
		return self.aug.augment_image(img)
		
class CasinoImgTransform:
	def __init__(self):
		self.aug = imgaug.augmenters.Sequential([
			imgaug.augmenters.Sometimes(0.4, imgaug.augmenters.CoarseDropout((0.01, 0.05), size_percent=(0.10, 0.25))),
			imgaug.augmenters.Sometimes(0.75, imgaug.augmenters.Affine(
				scale = (0.7, 1.2),
				translate_percent = {"x": (-0.35, 0.35), "y": (-0.35, 0.35)},
				rotate = (0, 360),
				order = 1,
				cval = (0, 255),
			)),
			#imgaug.augmenters.Sometimes(0.25, imgaug.augmenters.GaussianBlur(sigma=[1.0, 1.8])),
			imgaug.augmenters.Sometimes(0.5, imgaug.augmenters.Add((0, 100))),
			imgaug.augmenters.Sometimes(0.5, imgaug.augmenters.Multiply((0.9, 1.5))),
			imgaug.augmenters.Sometimes(0.3, imgaug.augmenters.Grayscale([0.5, 1.0])),
			imgaug.augmenters.AddToHueAndSaturation((-15, 15)),
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
	"xwing_red_gd": {
		"hsv_ranges": [XWING_RED_GD_DIE_HSV_RANGE_1, XWING_RED_GD_DIE_HSV_RANGE_2],
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
	"casino_blue": {
		"hsv_ranges": [WHITE_DOTS_HSV_RANGE],
		"rect_width": 100,
		"rect_height": 100,
		"classes_count": 6,			# 1-6
		"expected_distribution": {"one": 1.0/6.0, "two": 1.0/6.0, "three": 1.0/6.0, "four": 1.0/6.0, "five": 1.0/6.0, "six": 1.0/6.0},
		"training": {
			"image_transform": CasinoImgTransform(),
			"lr": 0.01,
			"momentum": 0.9,
			"lr_reduction_steps": 40,
			"total_steps": 80,
		},
	},
	"d8_blue": {
		"hsv_ranges": [D8_BLUE_DIE_HSV_RANGE],
		"rect_width": 84,
		"rect_height": 84,
		"classes_count": 8,			# 1-8
		"expected_distribution": {"one": 1.0/8.0, "two": 1.0/8.0, "three": 1.0/8.0, "four": 1.0/8.0, "five": 1.0/8.0, "six": 1.0/8.0, "seven": 1.0/8.0, "eight": 1.0/8.0},
		"training": {
			"image_transform": XwingImgTransform(),		# TODO
			"lr": 0.01,
			"momentum": 0.9,
			"lr_reduction_steps": 30,
			"total_steps": 120,
		},
	},
}
