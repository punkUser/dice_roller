# Settings
INPUT_FILE = 'output/captured_data/test4/20181222_113132/D/classified/dice.csv'

XWING_RED_EXPECTED_DIST = {"blank": 2.0/8.0, "focus": 2.0/8.0, "hit":   3.0/8.0, "crit":  1.0/8.0}
XWING_GREEN_EXPECTED_DIST = {"blank": 3.0/8.0, "focus": 2.0/8.0, "evade": 3.0/8.0}
CASINO_EXPECTED_DIST = {"one": 1.0/6.0, "two": 1.0/6.0, "three": 1.0/6.0, "four": 1.0/6.0, "five": 1.0/6.0, "six": 1.0/6.0}
# TODO: Improve this
EXPECTED_DIST = CASINO_EXPECTED_DIST

###################################################################################################
import numpy as np
import matplotlib.pyplot as plt
import csv

def plot_data_subset(title, roll_subplot, chisq_subplot, start=0, count=-1, stride=1):
	raw_start = 1 + start
	raw_end = -1 if count < 0 else raw_start + count
	data = np.array(g_raw_data)[raw_start:raw_end:stride,1:].astype(np.float)
	labels = g_raw_data[0][1:]
	plot(title, roll_subplot, chisq_subplot, data, labels)
	
def plot(title, roll_subplot, chisq_subplot, data, labels):
	# First dimension is roll #, second is dice label
	# Cumulative totals of all rolls up until that point
	data_totals = np.cumsum(data, 0)

	x = np.arange(1, data_totals.shape[0] + 1, 1)
	
	expected = np.zeros_like(data_totals)
	for i, label in enumerate(labels):
		expected[:,i] = EXPECTED_DIST[label] * x
	
	error_squared = np.square(data_totals - expected) / expected
	chi_squared = np.sum(error_squared, 1)
		
	# Roll distribution
	roll_plot = g_fig.add_subplot(roll_subplot)
	colors = ['b', 'y', 'g', 'r', 'c', 'm', 'k']
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
	chisq_plot.set_ylim(bottom=0, top=20)
	chisq_plot.set_xlabel('roll')
	chisq_plot.set_ylabel('Chi Squared')
	#chisq_plot.set_title('Chi Squared')
	chisq_plot.grid()
	
if __name__ == "__main__":
	with open(INPUT_FILE, newline='') as csvfile:
		g_raw_data = list(csv.reader(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL))
	
	g_fig = plt.figure(figsize=(20, 10))
	plot_data_subset("Roll Distribution", 231, 234, 0, -1, 1)
	plot_data_subset("Even Rolls",        232, 235, 0, -1, 2)
	plot_data_subset("Odd Rolls",         233, 236, 1, -1, 2)
		
	plt.tight_layout()
	plt.show()