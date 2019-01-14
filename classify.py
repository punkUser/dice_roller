import torchvision.datasets
import torchvision.transforms
import torch.utils.data
import numpy as np
import os
import PIL
import shutil
import csv
from pathlib import Path
import dice_cnn
import die_types

# Settings
ROOT_DATA_DIR  = 'output/captured_data/xr1_cd19_cd20_xg10/20190113_150310/B/'
DIE_TYPE = "casino_blue"
COPY_CLASSIFIED_FILES = True
INPUT_EXT = '.jpg'

###################################################################################################

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
	test_loader	= torch.utils.data.DataLoader(test_set,	batch_size=256, shuffle=False, num_workers=4)

	print("Input set size: {}".format(len(test_set)))
	
	# DEBUG
	images = iter(test_loader).next()
	dice_cnn.show_tensor_image(torchvision.utils.make_grid(images[0][0:16], nrow = 4))
		
	# TODO: Sort out this workaround for class label timing... dependency is only on the # of classes really
	model = dice_cnn.Model([str(x) for x in range(die_types.params[DIE_TYPE]["classes_count"])], die_types.params[DIE_TYPE]["rect_size"])
	model.load(os.path.join("models", DIE_TYPE + ".tar"))
	class_labels = model.get_class_labels()
	
	# Ensure output directories exist, but delete any image files (to avoid merging with previous run data)
	output_dir = os.path.join(ROOT_DATA_DIR, 'classified')
	if COPY_CLASSIFIED_FILES:
		for class_label in class_labels:
			path = os.path.join(output_dir, class_label)
			if not os.path.exists(path):
				os.makedirs(path)
			else:
				# Remove any image files first
				for file in Path(path).glob('*' + INPUT_EXT):
					os.remove(file)
				

	# TODO Maybe give it a name based on the path instead so it could be moved to same dir as others
	csv_file = open(os.path.join(output_dir, "dice.csv"), mode='w', newline='')
	csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

	csv_writer.writerow(["File"] + class_labels)
	
	for images, _unused, paths in test_loader:
		labels = model.classify(images)
		for label, path in zip(labels, paths):
			labels = [0] * len(class_labels)
			labels[label] = 1
			csv_writer.writerow([os.path.basename(path)] + labels)
		
			if COPY_CLASSIFIED_FILES:
				target_dir = os.path.join(output_dir, class_labels[label])
				shutil.copy2(path, target_dir)

if __name__ == "__main__":
	main()