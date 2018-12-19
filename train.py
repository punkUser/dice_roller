# Settings
TRAIN_DATA = 'training_data/xwing_red'
TEST_DATA  = 'test_data/xwing_red'

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
	img = img / 2 + 0.5     # unnormalize
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

class SimpleNet(nn.Module):
	def __init__(self, num_classes):
		super(SimpleNet,self).__init__()

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

def save_models(epoch):
	#torch.save(model.state_dict(), "cifar10model_{}.model".format(epoch))
	#print("Checkpoint saved")
	pass

def test(showErrorImages = False):
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
		
		# Debug which ones are failing
		if showErrorImages:
			for i in range(len(predicted)):
				if predicted[i] != labels[i]:
					print("Predicted {}, expected {}. Weights {}".format(classLabelStrings[predicted[i]], classLabelStrings[labels[i]], outputs.data[i]))
					imshow(images[i].cpu())

	test_acc = correct / total
	return test_acc

def train(num_epochs):
	best_acc = 0

	for epoch in range(num_epochs):
		# Update epoch-based optimizer learning rate
		scheduler.step()
		
		model.train()
		
		total = 0
		train_acc = 0
		train_loss = 0
		for i, (images, labels) in enumerate(trainLoader):
			# Move images and labels to gpu if available
			images, labels = images.to(device), labels.to(device)

			optimizer.zero_grad()
			outputs = model(images)
			loss = loss_fn(outputs,labels)
			loss.backward()

			optimizer.step()

			train_loss += loss.item() * images.size(0)
			
			total += labels.size(0)
			_, predicted = torch.max(outputs.data, 1)
			train_acc += (predicted == labels).sum().item()

		# Compute the accuracy and loss
		train_acc = train_acc / total
		train_loss = train_loss / total

		test_acc = test()

		# Save the model if the test acc is greater than our current best
		if test_acc > best_acc:
			save_models(epoch)
			best_acc = test_acc

		print("Epoch {}, Train Accuracy: {} , TrainLoss: {} , Test Accuracy: {:.5f}".format(epoch, train_acc, train_loss, test_acc))

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
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print(device)
	
	# Training data
	trainTransform = torchvision.transforms.Compose([
		ImgAugTrainTransform(),
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])
	
	trainSet = torchvision.datasets.ImageFolder(root=TRAIN_DATA, transform=trainTransform)	
	print("Raw training set size: {}".format(len(trainSet)))
	trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
	
	classLabelStrings = trainSet.classes
	print("Classes: {}".format(classLabelStrings))
	
	# Test data
	testTransform = torchvision.transforms.Compose([
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])
	testSet = torchvision.datasets.ImageFolder(root=TEST_DATA, transform=testTransform)	
	print("Test set size: {}".format(len(testSet)))
	testLoader  = torch.utils.data.DataLoader(testSet,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


	#trainCount = int(0.8 * datasetCount)
	#trainSet, testSet = torch.utils.data.random_split(dataset, [trainCount, datasetCount - trainCount])
	
	# DEBUG: Show some images
	images, labels = iter(trainLoader).next()
	print(' '.join('%5s' % trainSet.classes[labels[j]] for j in range(4)))
	imshow(torchvision.utils.make_grid(images[0:16], nrow = 4))
		
	# Create model, optimizer and loss function
	model = SimpleNet(num_classes = len(trainSet.classes))
	model.to(device)

	optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20,  gamma = 0.1)
	loss_fn = nn.CrossEntropyLoss()

	train(20)
	
	# Final test and display of mispredicted ones
	test_acc = test(True)