# Settings
TRAIN_DATA = 'training_data/xwing_red'
TEST_DATA  = 'test_data/xwing_red'
OUTPUT_MODEL_FILE = 'output/xwing_red.tar'

# TODO: Play with this - probably want to resize before dumping into the NN
IMAGE_DIMENSIONS = 84
# NOTE: Affects batch norm as well, so generally should be at least 8 or 16 or so for current network
BATCH_SIZE = 16


###################################################################################################
import torch
import torch.nn as nn
import torch.optim
import torchvision.datasets
import torchvision.transforms
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa
import imgaug

def imshow(img):
	img = img / 2 + 0.5		# unnormalize
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.show()

class ConvUnit(nn.Module):
	def __init__(self, in_channels, out_channels, stride=1, padding=1):
		super(ConvUnit, self).__init__()		
		self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=padding)
		self.relu = nn.ReLU()
		self.bn = nn.BatchNorm2d(num_features=out_channels)

	def forward(self,input):
		output = self.conv(input)
		output = self.relu(output)
		output = self.bn(output)
		return output

class Net(nn.Module):
	def __init__(self, num_classes):
		super(Net, self).__init__()

		# NOTE: Probably a overkill network for our problem but it's fast and it works,
		# so not much motivation to optimize it down at the moment.
		
		self.unit1 = ConvUnit(in_channels=3, out_channels=4)
		self.unit2 = ConvUnit(in_channels=4, out_channels=4)
		self.unit3 = ConvUnit(in_channels=4, out_channels=4)

		# In some ways letting the network learn the pooling step via strided convolution is nice,
		# but in practice MaxPool is somewhat quicker and more consistent for our data set right now.
		self.pool1 = nn.MaxPool2d(kernel_size=2)		
		#self.pool1 = ConvUnit(in_channels=4, out_channels=4, stride=2)
		
		#self.pool1 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=2, padding=1)
		#self.pool1relu = nn.ReLU();

		self.unit4 = ConvUnit(in_channels=4, out_channels=8)
		self.unit5 = ConvUnit(in_channels=8, out_channels=8)
		self.unit6 = ConvUnit(in_channels=8, out_channels=8)
		self.unit7 = ConvUnit(in_channels=8, out_channels=8)

		self.pool2 = nn.MaxPool2d(kernel_size=2)
		#self.pool2 = ConvUnit(in_channels=8, out_channels=8, stride=2)
		
		# Add all the units into the Sequential layer in exact order
		self.net = nn.Sequential(self.unit1, self.unit2, self.unit3, self.pool1,
								 self.unit4, self.unit5, self.unit6, self.unit7, self.pool2)

		 # Two 1/2 size pooling steps
		dimAfterPooling = int(IMAGE_DIMENSIONS / 4)
		self.fcSize = 8 * dimAfterPooling * dimAfterPooling
		
		self.fc = nn.Linear(in_features=self.fcSize, out_features=num_classes)

	def forward(self, input):
		output = self.net(input)
		#print(output.shape)
		output = output.view(-1, self.fcSize)
		output = self.fc(output)
		return output

class Model:
	def __init__(self, classLabelStrings):
		self.classLabelStrings = classLabelStrings
	
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		print("Device: {}".format(self.device))
					
		# Create model, optimizer and loss function
		self.model = Net(num_classes = len(self.classLabelStrings))
		self.model.to(self.device)

		self.optimizer    = torch.optim.SGD(self.model.parameters(), lr = 0.01, momentum = 0.9)
		self.scheduler    = torch.optim.lr_scheduler.StepLR(self.optimizer, 20,	gamma = 0.1)
		self.lossFunction = nn.CrossEntropyLoss()
			
	def save(self, fileName = OUTPUT_MODEL_FILE):
		torch.save({
			'epoch': self.epoch,
			'model_state_dict': self.model.state_dict(),
			'optimizer_state_dict': self.optimizer.state_dict(),
			'scheduler_state_dict': self.scheduler.state_dict(),
			'loss': self.lossFunction,
			}, fileName)
		#print('Model saved to {}'.format(OUTPUT_MODEL_FILE))
		
	def load(self, fileName = OUTPUT_MODEL_FILE):
		checkpoint = torch.load(OUTPUT_MODEL_FILE)
		
		self.epoch = checkpoint['epoch']
		self.model.load_state_dict(checkpoint['model_state_dict'])
		self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])		
		self.lossFunction = checkpoint['loss']

	def test(self, loader, showErrorImages = False):
		self.model.eval()
		correct = 0
		total = 0
		for i, (images, labels) in enumerate(loader):
			images, labels = images.to(self.device), labels.to(self.device)

			# Predict classes using images from the test set
			outputs = self.model(images)
			_,predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
			
			# Debug which ones are failing
			if showErrorImages:
				# DEBUG
				#for i in range(len(predicted)):
				#	maxRow = torch.max(outputs.data[i])
				#	row = outputs.data[i] / maxRow
				#	print("{}, max {}".format(row, maxRow))
					
				for i in range(len(predicted)):
					if predicted[i] != labels[i]:
						print("Index ~{} Predicted {}, expected {}. Weights {}".format(total, classLabelStrings[predicted[i]], classLabelStrings[labels[i]], outputs.data[i]))
						imshow(images[i].cpu())

		test_acc = correct / total
		return test_acc
		
	def train(self, numEpochs, trainLoader, testLoader):		
		for epoch in range(numEpochs):
			self.epoch = epoch
			
			# Update epoch-based optimizer learning rate
			self.scheduler.step()
			self.model.train()
			
			total = 0
			trainAcc = 0
			trainLoss = 0
			for i, (images, labels) in enumerate(trainLoader):
				# Move images and labels to gpu if available
				images, labels = images.to(self.device), labels.to(self.device)

				self.optimizer.zero_grad()
				outputs = self.model(images)
				loss = self.lossFunction(outputs,labels)
				loss.backward()

				self.optimizer.step()

				trainLoss += loss.item() * images.size(0)
				
				total += labels.size(0)
				_, predicted = torch.max(outputs.data, 1)
				trainAcc += (predicted == labels).sum().item()

			# Compute the accuracy and loss
			trainAcc = trainAcc / total
			trainLoss = trainLoss / total

			testAcc = self.test(testLoader)
			print("Epoch {}, Train Accuracy: {:.5f} , TrainLoss: {} , Test Accuracy: {:.5f}".format(epoch, trainAcc, trainLoss, testAcc))
			
			# Save checkpoint
			self.save()

class ImgAugTrainTransform:
	def __init__(self):
		self.aug = iaa.Sequential([
			iaa.Affine(
				scale = {"x": (0.9, 1.1), "y": (0.9, 1.1)},
				translate_percent = {"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
				rotate = (0, 360),
				order = 1,
				cval = (0, 255),
			),
			iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=[1.0, 1.8])),
			iaa.AddToHueAndSaturation((-10, 10)),
			#iaa.AdditiveGaussianNoise(loc = 0, scale = (0.0, 0.05*255), per_channel = 0.5)
		])

	def __call__(self, img):
		img = np.array(img)
		return self.aug.augment_image(img)


if __name__ == "__main__":	
	# Training data
	trainTransform = torchvision.transforms.Compose([
		ImgAugTrainTransform(),
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])	
	trainSet = torchvision.datasets.ImageFolder(root=TRAIN_DATA, transform=trainTransform)
	trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
	
	# Test data
	testTransform = torchvision.transforms.Compose([
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])
	testSet = torchvision.datasets.ImageFolder(root=TEST_DATA, transform=testTransform)	
	testLoader	= torch.utils.data.DataLoader(testSet,	batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

	classLabelStrings = trainSet.classes
	print("Classes: {}".format(classLabelStrings))
	print("Raw training set size: {}".format(len(trainSet)))
	print("Test set size: {}".format(len(testSet)))
	
	# DEBUG: Show some images
	images, labels = iter(trainLoader).next()
	print(' '.join('%5s' % classLabelStrings[labels[j]] for j in range(4)))
	imshow(torchvision.utils.make_grid(images[0:16], nrow = 4))
		
	diceModel = Model(classLabelStrings)
	diceModel.load() # Continue onwards!
	#diceModel.train(30, trainLoader, testLoader)
	
	# Final test and display of mispredicted ones
	testAcc = diceModel.test(testLoader, True)
	print("Test Set Accuracy: {}".format(testAcc))
	
	# Now test all the training ones too, but without the augmentation
	rawTrainSet = torchvision.datasets.ImageFolder(root=TRAIN_DATA, transform=testTransform)
	rawTrainLoader = torch.utils.data.DataLoader(rawTrainSet, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
	rawTrainAcc = diceModel.test(rawTrainLoader, True)
	print("Raw trainT Set Accuracy: {}".format(rawTrainAcc))
	