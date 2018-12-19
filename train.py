# TODO: Play with this - probably want to resize before dumping into the NN
IMAGE_DIMENSIONS = 84


import torch
import torch.nn as nn
import torch.optim
import torchvision.datasets
import torchvision.transforms
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt

def imshow(img):
	img = img / 2 + 0.5     # unnormalize
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.show()


class Unit(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(Unit, self).__init__()		
		self.conv = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, stride = 1, padding = 1)
		#self.bn = nn.BatchNorm2d(num_features = out_channels)
		self.relu = nn.ReLU()

	def forward(self,input):
		output = self.conv(input)
		#output = self.bn(output)
		output = self.relu(output)
		return output

class SimpleNet(nn.Module):
	def __init__(self, num_classes):
		super(SimpleNet,self).__init__()

		self.unit1 = Unit(in_channels= 3, out_channels=32)
		self.unit2 = Unit(in_channels=32, out_channels=32)
		self.unit3 = Unit(in_channels=32, out_channels=32)

		self.pool1 = nn.MaxPool2d(kernel_size = 2)

		self.unit4 = Unit(in_channels=32, out_channels=64)
		self.unit5 = Unit(in_channels=64, out_channels=64)
		self.unit6 = Unit(in_channels=64, out_channels=64)
		self.unit7 = Unit(in_channels=64, out_channels=64)

		self.pool2 = nn.MaxPool2d(kernel_size = 2)
		
		# Add all the units into the Sequential layer in exact order
		#self.net = nn.Sequential(self.unit1, self.unit2, self.unit3, self.pool1,
		#						 self.unit4, self.unit5, self.unit6, self.unit7, self.pool2)
		
		self.net = nn.Sequential(self.unit1, self.unit2, self.unit3, self.pool1)

		# Image expected to be 84*84
		self.fc = nn.Linear(in_features = int(32/4) * IMAGE_DIMENSIONS * IMAGE_DIMENSIONS, out_features = num_classes)

	def forward(self, input):
		output = self.net(input)
		output = output.view(-1, int(32/4) * IMAGE_DIMENSIONS * IMAGE_DIMENSIONS)
		output = self.fc(output)
		return output

def save_models(epoch):
	#torch.save(model.state_dict(), "cifar10model_{}.model".format(epoch))
	#print("Checkpoint saved")
	pass

def test():
	model.eval()
	correct = 0
	total = 0
	for i, (images, labels) in enumerate(testLoader):	
		images, labels = images.to(device), labels.to(device)

		# Predict classes using images from the test set
		outputs = model(images)
		_,predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()

	# Compute the average acc and loss over all 10000 test images
	test_acc = correct / total
	return test_acc

def train(num_epochs):
	best_acc = 0

	for epoch in range(num_epochs):
		scheduler.step()
		model.train()
		total = 0
		train_acc = 0
		train_loss = 0
		for i, (images, labels) in enumerate(trainLoader):
			# Move images and labels to gpu if available
			images, labels = images.to(device), labels.to(device)

			# Clear all accumulated gradients
			optimizer.zero_grad()
			# Predict classes using images from the test set
			outputs = model(images)
			# Compute the loss based on the predictions and actual labels
			loss = loss_fn(outputs,labels)
			# Backpropagate the loss
			loss.backward()

			# Adjust parameters according to the computed gradients
			optimizer.step()

			train_loss += loss.item() * images.size(0)
			
			total += labels.size(0)
			_, predicted = torch.max(outputs.data, 1)
			train_acc += (predicted == labels).sum().item()

		# Compute the average acc and loss over all 50000 training images
		train_acc = train_acc / total
		train_loss = train_loss / total

		# Evaluate on the test set
		test_acc = test()

		# Save the model if the test acc is greater than our current best
		if test_acc > best_acc:
			save_models(epoch)
			best_acc = test_acc

		# Print the metrics
		print("Epoch {}, Train Accuracy: {} , TrainLoss: {} , Test Accuracy: {}".format(epoch, train_acc, train_loss,test_acc))


if __name__ == "__main__":
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print(device)

	# TODO: Sort out if we need different transformations for training/test
	transform = torchvision.transforms.Compose([
		#transforms.RandomHorizontalFlip(),
		#transforms.RandomCrop(32, padding = 4),
		torchvision.transforms.ColorJitter(0.2, 0.2, 0.05, 0.05),
		torchvision.transforms.RandomRotation(180),
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])

	batch_size = 16

	dataset = torchvision.datasets.ImageFolder(root='training_data/xwing_red', transform = transform)	
	datasetCount = len(dataset)

	trainCount = int(0.8 * datasetCount)
	trainSet, testSet = torch.utils.data.random_split(dataset, [trainCount, datasetCount - trainCount])
	
	trainLoader = torch.utils.data.DataLoader(trainSet, batch_size = batch_size, shuffle = True, num_workers = 4)
	testLoader  = torch.utils.data.DataLoader(testSet,  batch_size = batch_size, shuffle = True, num_workers = 4)
	
	# DEBUG: Show some images
	images, labels = iter(trainLoader).next()
	print(' '.join('%5s' % dataset.classes[labels[j]] for j in range(4)))
	imshow(torchvision.utils.make_grid(images[0:16], nrow = 4))
		
	# Create model, optimizer and loss function
	model = SimpleNet(num_classes = len(dataset.classes))
	model.to(device)

	# 0.0001 default learning rate
	optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30,  gamma = 0.1)
	loss_fn = nn.CrossEntropyLoss()

	train(50)