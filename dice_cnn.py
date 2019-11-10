import torch
import torch.nn as nn
import torch.optim
import torchvision.datasets
import numpy as np
import matplotlib.pyplot as plt

# Add paths to the return value of ImageFolders; useful for user output/messages
class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

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
	# TODO: Support rectangular images?
	def __init__(self, classes_count, image_width, image_height):
		super(Net, self).__init__()

		# NOTE: Probably a overkill network for our problem but it's fast and it works,
		# so not much motivation to optimize it down at the moment.
		
		self.unit1 = ConvUnit(in_channels=3, out_channels=8)
		self.unit2 = ConvUnit(in_channels=8, out_channels=8)
		self.unit3 = ConvUnit(in_channels=8, out_channels=8)

		# In some ways letting the network learn the pooling step via strided convolution is nice,
		# but in practice MaxPool is somewhat quicker and more consistent for our data set right now.
		self.pool1 = nn.MaxPool2d(kernel_size=2)		
		#self.pool1 = ConvUnit(in_channels=4, out_channels=4, stride=2)
		
		#self.pool1 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=2, padding=1)
		#self.pool1relu = nn.ReLU();

		self.unit4 = ConvUnit(in_channels=8, out_channels=16)
		self.unit5 = ConvUnit(in_channels=16, out_channels=16)
		self.unit6 = ConvUnit(in_channels=16, out_channels=16)
		self.unit7 = ConvUnit(in_channels=16, out_channels=16)

		self.pool2 = nn.MaxPool2d(kernel_size=2)
		#self.pool2 = ConvUnit(in_channels=8, out_channels=8, stride=2)
		
		# Add all the units into the Sequential layer in exact order
		self.net = nn.Sequential(self.unit1, self.unit2, self.unit3, self.pool1,
								 self.unit4, self.unit5, self.unit6, self.unit7, self.pool2)

		 # Two 1/2 size pooling steps
		widthAfterPooling = int(image_width / 4)
		heightAfterPooling = int(image_height / 4)
		self.fcSize = 16 * widthAfterPooling * heightAfterPooling
		
		self.fc = nn.Linear(in_features=self.fcSize, out_features=classes_count)

	def forward(self, input):
		output = self.net(input)
		#print(output.shape)
		output = output.view(-1, self.fcSize)
		output = self.fc(output)
		return output

class Model:
	def __init__(self, class_labels, image_width, image_height, lr = 0.01, momentum = 0.9, lr_reduction_steps = 20):
		self.class_labels = class_labels
		self.epoch = 0
	
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		print("Device: {}".format(self.device))
		#print(torch.cuda.current_device())
		#print(torch.cuda.device(0))
		#print(torch.cuda.device_count())
		#print(torch.cuda.get_device_name(0))
					
		# Create model, optimizer and loss function
		self.model = Net(len(self.class_labels), image_width, image_height)
		self.model.to(self.device)

		self.optimizer    = torch.optim.SGD(self.model.parameters(), lr = lr, momentum = momentum)
		self.scheduler    = torch.optim.lr_scheduler.StepLR(self.optimizer, lr_reduction_steps,	gamma = 0.1)
		self.loss_function = nn.CrossEntropyLoss()

	def get_class_labels(self):
		return self.class_labels
		
	def save(self, file_name):
		torch.save({
			'epoch': self.epoch,
			'model_state_dict': self.model.state_dict(),
			'optimizer_state_dict': self.optimizer.state_dict(),
			'scheduler_state_dict': self.scheduler.state_dict(),
			'loss': self.loss_function,
			'class_labels': self.class_labels,
			}, file_name)
		print('Model saved to {}'.format(file_name))
		
	def load(self, file_name):
		checkpoint = torch.load(file_name)
		
		self.epoch = checkpoint['epoch']
		self.model.load_state_dict(checkpoint['model_state_dict'])
		self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])		
		self.loss_function = checkpoint['loss']
		self.class_labels = checkpoint['class_labels']
		
		# TODO: Sort out train GPU vs infer CPU and other combos...
		
		print('Model loaded from {}'.format(file_name))

	# Classify a batch of images with the model, returning predicted class tensor
	def classify(self, images):
		self.model.eval()
		images = images.to(self.device)

		# Predict classes using images from the test set
		outputs = self.model(images)
		_, predicted = torch.max(outputs.data, 1)
		return predicted
		
	# TODO: Pull loader loop outside of this class likely
	def test(self, loader, show_error_images = False):
		self.model.eval()
		correct = 0
		total = 0
		for images, labels, path in loader:
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
						print("{}: Predicted {}, expected {}. Weights {}".format(path[i], self.class_labels[predicted[i]], self.class_labels[labels[i]], outputs.data[i]))
						show_tensor_image(images[i].cpu())

		test_acc = correct / total
		return test_acc
	
	# TODO: Pull loader loop outside of this class likely
	def train(self, numEpochs, train_loader, test_loader):
		for localEpoch in range(numEpochs):
			self.model.train()
			
			total = 0
			train_accuracy = 0
			train_loss = 0
			for images, labels, path in train_loader:
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
			
			# Update epoch-based optimizer learning rate
			self.scheduler.step()
			
			#self.save("output/checkpoint.tar")
			self.epoch += 1
