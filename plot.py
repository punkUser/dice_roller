import numpy as np
import matplotlib.pyplot as plt
import csv
import die_types
import os.path;

# Settings
INPUT_FILE = "results/d8_blue/d8b4_run1_compartmentD.csv"
DIE_TYPE = "d8_blue"

###################################################################################################

def plot(title, roll_subplot, chisq_subplot, data, labels, print_totals = False):
	# First dimension is roll #, second is dice label
	# Cumulative totals of all rolls up until that point
	data = np.array(data)[:,1:].astype(np.float)
	data_totals = np.cumsum(data, 0)

	x = np.arange(1, data_totals.shape[0] + 1, 1)
		
	expected = np.zeros_like(data_totals)
	expected_distribution = die_types.params[DIE_TYPE]["expected_distribution"]
	for i, label in enumerate(labels):
		expected[:,i] = expected_distribution[label] * x
	
	error_squared = np.square(data_totals - expected) / expected
	chi_squared = np.sum(error_squared, 1)
	
	if print_totals:
		print("{}".format(INPUT_FILE))
		print("total rolls: {}".format(data_totals.shape[0]))
		print("chi squared: {}".format(chi_squared[-1]))
		for i, label in enumerate(labels):
			print("{}: Got {}, expected {} (delta {})".format(label, data_totals[-1,i], expected[-1,i], data_totals[-1,i] - expected[-1,i]))
		
	# Roll distribution
	roll_plot = g_fig.add_subplot(roll_subplot)
	colors = ['b', 'y', 'g', 'r', 'c', 'm', 'k', 'w']
	for i, label in enumerate(labels):
		roll_plot.plot(x, data_totals[:,i], colors[i] + '-', label="total " + label)
		roll_plot.plot(x,    expected[:,i], colors[i] + ':', label="expected " + label)
	roll_plot.set_xlim(left=1, right=x[-1])
	roll_plot.set_ylim(bottom=0)
	roll_plot.set_xlabel('roll')
	roll_plot.set_ylabel('total')
	roll_plot.set_title(title)
	roll_plot.legend()
	roll_plot.grid()
	
	chisq_plot = g_fig.add_subplot(chisq_subplot)
	chisq_plot.plot(x, chi_squared, 'b-')
	chisq_plot.set_xlim(left=1, right=x[-1])
	#chisq_plot.set_ylim(bottom=0, top=20)
	chisq_plot.set_xlabel('roll')
	chisq_plot.set_ylabel('Chi Squared')
	#chisq_plot.set_title('Chi Squared')
	chisq_plot.grid()
	
if __name__ == "__main__":
	with open(INPUT_FILE, newline='') as csvfile:
		raw_data = list(csv.reader(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL))
	labels = raw_data[0][1:]
	data = raw_data[1:]
	
	g_fig = plt.figure(figsize=(15, 8))
	plot("Roll Distribution", 211, 212, data[0:-1:1], labels, True)
	#plot("Even Rolls",        232, 235, data[0:-1:2], labels)
	#plot("Odd Rolls",         233, 236, data[1:-1:2], labels)
	
	# Show distribution of rolls that immediately follows a given roll
	#rolls_following_plot_count = min(3, len(labels))
	#g_fig = plt.figure(figsize=(20, 10))
	#subplot_base = 200 + 10 * rolls_following_plot_count
	#for label_index, label in enumerate(labels[0:rolls_following_plot_count]):
	#	roll_subplot = subplot_base + label_index + 1
	#	chisq_subplot = roll_subplot + rolls_following_plot_count
	#	rolls_following_label = [x for i, x in enumerate(data) if i > 1 and int(data[i-1][label_index+1]) > 0]
	#	plot("Distribution after {}".format(label), roll_subplot, chisq_subplot, rolls_following_label, labels)
		
	plt.tight_layout()
	g_fig.savefig(os.path.splitext(INPUT_FILE)[0] + '.png')
	plt.show()