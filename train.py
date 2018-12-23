# Settings
DATA = "casino_blue"

# NOTE: Affects batch norm as well, so generally should be at least 8 or 16 or so for training
BATCH_SIZE = 16


###################################################################################################
import dice_cnn

import torchvision.datasets
import torchvision.transforms
import torch.utils.data
import numpy as np
import os
import imgaug

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
				scale = (0.8, 1.1),
				translate_percent = {"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
				rotate = (0, 360),
				order = 1,
				cval = (0, 255),
			)),
			imgaug.augmenters.Sometimes(0.25, imgaug.augmenters.GaussianBlur(sigma=[1.0, 1.8])),
			imgaug.augmenters.AddToHueAndSaturation((-5, 5)),
			#imgaug.augmenters.AdditiveGaussianNoise(loc = 0, scale = (0.0, 0.05*255), per_channel = 0.5)
		])

	def __call__(self, img):
		img = np.array(img)
		return self.aug.augment_image(img)

def main():
	# Training data
	train_transform = torchvision.transforms.Compose([
		CasinoImgTransform(),
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])	
	train_set = dice_cnn.ImageFolderWithPaths(root=os.path.join("training_data", DATA), transform=train_transform)
	train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
	
	# Test data
	test_transform = torchvision.transforms.Compose([
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])
	test_set = dice_cnn.ImageFolderWithPaths(root=os.path.join("test_data", DATA), transform=test_transform)	
	test_loader	= torch.utils.data.DataLoader(test_set,	batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

	class_label_strings = train_set.classes
	print("Classes: {}".format(class_label_strings))
	print("Raw training set size: {}".format(len(train_set)))
	print("Test set size: {}".format(len(test_set)))
	
	# DEBUG: Show some training images
	images, labels, paths = iter(train_loader).next()
	dice_cnn.show_tensor_image(torchvision.utils.make_grid(images[0:16], nrow = 4))
	
	# Use the dimensions of the first image as representative	
	image_dimensions = images[0].shape[1]
	model_output_file = os.path.join("output", DATA + ".tar")
	model = dice_cnn.Model(class_label_strings, image_dimensions)
	
	#model.load(model_output_file) # Continue onwards!
	
	model.train(30, train_loader, test_loader)
	model.save(model_output_file)
	model.train(30, train_loader, test_loader)
	model.save(model_output_file)
	#model.train(60, train_loader, test_loader)
	#model.save(model_output_file)
	
	# Final test and display of mispredicted ones
	test_accuracy = model.test(test_loader, True)
	print("Test Set Accuracy: {}".format(test_accuracy))
	
	# Now test all the training ones too, but without the augmentation
	raw_train_set = dice_cnn.ImageFolderWithPaths(root=os.path.join("training_data", DATA), transform=test_transform)
	raw_train_loader = torch.utils.data.DataLoader(raw_train_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
	raw_train_acc = model.test(raw_train_loader, True)
	print("Raw trainT Set Accuracy: {}".format(raw_train_acc))

if __name__ == "__main__":
	main()