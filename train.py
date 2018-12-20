# Settings
TRAIN_DATA = 'training_data/xwing_red'
TEST_DATA  = 'test_data/xwing_red'
OUTPUT_MODEL_FILE = 'output/xwing_red.tar'

IMAGE_DIMENSIONS = 84
# NOTE: Affects batch norm as well, so generally should be at least 8 or 16 or so for training
BATCH_SIZE = 16


###################################################################################################
import dice_cnn

import torchvision.datasets
import torchvision.transforms
import torch.utils.data
import numpy as np
import imgaug

class ImgAugtrain_transform:
	def __init__(self):
		self.aug = imgaug.augmenters.Sequential([
			imgaug.augmenters.Affine(
				scale = {"x": (0.9, 1.1), "y": (0.9, 1.1)},
				translate_percent = {"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
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

def main():
	# Training data
	train_transform = torchvision.transforms.Compose([
		ImgAugtrain_transform(),
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])	
	train_set = torchvision.datasets.ImageFolder(root=TRAIN_DATA, transform=train_transform)
	train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
	
	# Test data
	test_transform = torchvision.transforms.Compose([
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])
	test_set = torchvision.datasets.ImageFolder(root=TEST_DATA, transform=test_transform)	
	test_loader	= torch.utils.data.DataLoader(test_set,	batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

	class_label_strings = train_set.classes
	print("Classes: {}".format(class_label_strings))
	print("Raw training set size: {}".format(len(train_set)))
	print("Test set size: {}".format(len(test_set)))
	
	# DEBUG: Show some training images
	images, labels = iter(train_loader).next()
	dice_cnn.show_tensor_image(torchvision.utils.make_grid(images[0:16], nrow = 4))
		
	model = dice_cnn.Model(class_label_strings, IMAGE_DIMENSIONS)
	#model.load(OUTPUT_MODEL_FILE) # Continue onwards!
	
	model.train(30, train_loader, test_loader)
	model.save(OUTPUT_MODEL_FILE)
	model.train(30, train_loader, test_loader)
	model.save(OUTPUT_MODEL_FILE)
	#model.train(60, train_loader, test_loader)
	#model.save(OUTPUT_MODEL_FILE)
	
	# Final test and display of mispredicted ones
	test_accuracy = model.test(test_loader, True)
	print("Test Set Accuracy: {}".format(test_accuracy))
	
	# Now test all the training ones too, but without the augmentation
	raw_train_set = torchvision.datasets.ImageFolder(root=TRAIN_DATA, transform=test_transform)
	raw_train_loader = torch.utils.data.DataLoader(raw_train_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
	raw_train_acc = model.test(raw_train_loader, True)
	print("Raw trainT Set Accuracy: {}".format(raw_train_acc))

if __name__ == "__main__":
	main()