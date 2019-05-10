import torchvision.datasets
import torchvision.transforms
import torch.utils.data
import numpy as np
import os
import dice_cnn
import die_types

# Settings
DIE_TYPE = "d8_blue"

# NOTE: Affects batch norm as well, so generally should be at least 8 or 16 or so for training
BATCH_SIZE = 16

###################################################################################################

def main():
	# Training data, including any die-specific training transform
	train_transform = torchvision.transforms.Compose([
		die_types.params[DIE_TYPE]["training"]["image_transform"],
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])	
	train_set = dice_cnn.ImageFolderWithPaths(root=os.path.join("training_data", DIE_TYPE), transform=train_transform)
	train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
	
	# Test data
	test_transform = torchvision.transforms.Compose([
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])
	test_set = dice_cnn.ImageFolderWithPaths(root=os.path.join("test_data", DIE_TYPE), transform=test_transform)	
	test_loader	= torch.utils.data.DataLoader(test_set,	batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

	class_label_strings = train_set.classes
	print("Classes: {}".format(class_label_strings))
	print("Raw training set size: {}".format(len(train_set)))
	print("Test set size: {}".format(len(test_set)))
	
	# DEBUG: Show some training images
	images, labels, paths = iter(train_loader).next()
	dice_cnn.show_tensor_image(torchvision.utils.make_grid(images[0:6], nrow = 6))
	
	# Use the dimensions of the first image as representative
	model_output_file = os.path.join("output", DIE_TYPE + ".tar")
	model = dice_cnn.Model(class_label_strings, images[0].shape[2], images[0].shape[1],
						   lr = die_types.params[DIE_TYPE]["training"]["lr"],
						   momentum = die_types.params[DIE_TYPE]["training"]["momentum"],
						   lr_reduction_steps = die_types.params[DIE_TYPE]["training"]["lr_reduction_steps"],)
	
	model.load(model_output_file) # Continue onwards!
	model.train(die_types.params[DIE_TYPE]["training"]["total_steps"], train_loader, test_loader)
	model.save(model_output_file)
	
	# Final test and display of mispredicted ones
	test_accuracy = model.test(test_loader, True)
	print("Test set accuracy: {}".format(test_accuracy))
	
	# Now test all the training ones too, but without the augmentation
	raw_train_set = dice_cnn.ImageFolderWithPaths(root=os.path.join("training_data", DIE_TYPE), transform=test_transform)
	raw_train_loader = torch.utils.data.DataLoader(raw_train_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
	raw_train_acc = model.test(raw_train_loader, True)
	print("Raw training set accuracy: {}".format(raw_train_acc))

if __name__ == "__main__":
	main()