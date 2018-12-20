# Settings
ROOT_DATA_DIR  = 'output/test2/20181214_144327/C/'
INPUT_MODEL_FILE = 'models/xwing_red.tar'
INPUT_EXT = '.jpg'
COPY_CLASSIFIED_FILES = True

IMAGE_DIMENSIONS = 84
# NOTE: Affects batch norm as well, so generally should be at least 8 or 16 or so for training
BATCH_SIZE = 16


###################################################################################################
import dice_cnn

import torchvision.datasets
import torchvision.transforms
import torch.utils.data
import numpy as np
import os
import PIL
import shutil
from pathlib import Path

# From torchvision
def pil_loader(path):
    with open(path, 'rb') as f:
        img = PIL.Image.open(f)
        return img.convert('RGB')
	
class Dataset(torch.utils.data.Dataset):
	def __init__(self, root, transform=None):
		# Find all the relevant files
		self.image_paths = []
		for file in sorted(Path(root).glob('*' + INPUT_EXT)):
			self.image_paths.append(file)
		
		if len(self.image_paths) == 0:
			raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"))

		self.root = root
		self.loader = pil_loader
		self.transform = transform

	def __getitem__(self, index):
		path = self.image_paths[index]		
		image = self.loader(path)
		if self.transform is not None:
			image = self.transform(image)
		# TODO: Sort out unused label parameter... in there to make DataLoader happy atm
		return (image, 0, str(path))

	def __len__(self):
		return len(self.image_paths)

def main():
	# Test data
	test_transform = torchvision.transforms.Compose([
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])
	test_set = Dataset(root=os.path.join(ROOT_DATA_DIR, 'cropped'), transform=test_transform)
	test_loader	= torch.utils.data.DataLoader(test_set,	batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

	print("Input set size: {}".format(len(test_set)))
	
	# DEBUG
	images = iter(test_loader).next()
	dice_cnn.show_tensor_image(torchvision.utils.make_grid(images[0:16][0], nrow = 4))
		
	# TODO: Sort out this workaround for class label timing... dependency is only on the # of classes really
	model = dice_cnn.Model([str(x) for x in range(4)], IMAGE_DIMENSIONS)
	model.load(INPUT_MODEL_FILE)
	class_labels = model.get_class_labels()
	
	# Ensure output directories exist
	output_dir = os.path.join(ROOT_DATA_DIR, 'classified')
	if COPY_CLASSIFIED_FILES:
		for class_label in class_labels:
			path = os.path.join(output_dir, class_label)
			if not os.path.exists(path):
				os.makedirs(path)
	
	
	for images, _unused, paths in test_loader:
		labels = model.classify(images)
		
		if COPY_CLASSIFIED_FILES:
			for label, path in zip(labels, paths):
				target_dir = os.path.join(output_dir, class_labels[label])
				shutil.copy2(path, target_dir)

if __name__ == "__main__":
	main()