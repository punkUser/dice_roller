import torch
import torch.nn as nn
import torch.optim
import numpy as np
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa
import imgaug

def show_tensor_image(img):
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
	def __init__(self, num_classes, image_dimensions):
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
		dimAfterPooling = int(image_dimensions / 4)
		self.fcSize = 8 * dimAfterPooling * dimAfterPooling
		
		self.fc = nn.Linear(in_features=self.fcSize, out_features=num_classes)

	def forward(self, input):
		output = self.net(input)
		#print(output.shape)
		output = output.view(-1, self.fcSize)
		output = self.fc(output)
		return output

class Model:
	def __init__(self, class_label_strings, image_dimensions):
		self.class_label_strings = class_label_strings
		self.epoch = 0
	
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		print("Device: {}".format(self.device))
					
		# Create model, optimizer and loss function
		self.model = Net(len(self.class_label_strings), image_dimensions)
		self.model.to(self.device)

		self.optimizer    = torch.optim.SGD(self.model.parameters(), lr = 0.01, momentum = 0.9)
		self.scheduler    = torch.optim.lr_scheduler.StepLR(self.optimizer, 20,	gamma = 0.1)
		self.loss_function = nn.CrossEntropyLoss()
			
	def save(self, file_name):
		torch.save({
			'epoch': self.epoch,
			'model_state_dict': self.model.state_dict(),
			'optimizer_state_dict': self.optimizer.state_dict(),
			'scheduler_state_dict': self.scheduler.state_dict(),
			'loss': self.loss_function,
			}, file_name)
		#print('Model saved to {}'.format(file_name))
		
	def load(self, file_name):
		checkpoint = torch.load(file_name)
		
		self.epoch = checkpoint['epoch']
		self.model.load_state_dict(checkpoint['model_state_dict'])
		self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])		
		self.loss_function = checkpoint['loss']
		
		print('Model loaded from {}'.format(file_name))

	def test(self, loader, show_error_images = False):
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
			if show_error_images:
				# DEBUG
				#for i in range(len(predicted)):
				#	maxRow = torch.max(outputs.data[i])
				#	row = outputs.data[i] / maxRow
				#	print("{}, max {}".format(row, maxRow))
					
				for i in range(len(predicted)):
					if predicted[i] != labels[i]:
						print("Index ~{} Predicted {}, expected {}. Weights {}".format(total, class_label_strings[predicted[i]], class_label_strings[labels[i]], outputs.data[i]))
						show_tensor_image(images[i].cpu())

		test_acc = correct / total
		return test_acc
		
	def train(self, numEpochs, train_loader, test_loader):
		for localEpoch in range(numEpochs):			
			# Update epoch-based optimizer learning rate
			self.scheduler.step()
			self.model.train()
			
			total = 0
			train_accuracy = 0
			train_loss = 0
			for i, (images, labels) in enumerate(train_loader):
				# Move images and labels to gpu if available
				images, labels = images.to(self.device), labels.to(self.device)

				self.optimizer.zero_grad()
				outputs = self.model(images)
				loss = self.loss_function(outputs,labels)
				loss.backward()

				self.optimizer.step()

				train_loss += loss.item() * images.size(0)
				
				total += labels.size(0)
				_, predicted = torch.max(outputs.data, 1)
				train_accuracy += (predicted == labels).sum().item()

			# Compute the accuracy and loss
			train_accuracy = train_accuracy / total
			train_loss = train_loss / total

			test_accuracy = self.test(test_loader)
			print("Epoch {}, Train Accuracy: {:.5f} , train_loss: {} , Test Accuracy: {:.5f}".format(self.epoch, train_accuracy, train_loss, test_accuracy))
			
			# Save checkpoint
			self.save("output/checkpoint.tar")
			self.epoch += 1
