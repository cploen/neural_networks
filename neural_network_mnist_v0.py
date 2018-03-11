import numpy
#imports special package for sigmoid function
import scipy.special
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
#neural network class definition
class neuralNetwork:

	# initialize the neural network
	def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
		#set number of nodes in each input, hidden, and output layer
		self.inodes = inputnodes
		self.hnodes = hiddennodes
		self.onodes = outputnodes

		# create two link weight matrices using self.inodes, self.hnodes, 
		# and self.onodes to set correct size matrices wih and who
		# weights inside the arrays are w_i_j, where the link
		# is from node i to node j in the next layer
		# w11 w21
		# w12 w22 etc

		# these weights need to be updated with something more sophisticated
		self.wih = (numpy.random.rand(self.hnodes, self.inodes) - 0.5)
		self.who = (numpy.random.rand(self.onodes, self.hnodes) - 0.5)


		# learning rate
		self.lr = learningrate
		pass

		# activation function (sigmoid) 
		self.activation_function = lambda x: scipy.special.expit(x)


	# train the neural network
	def train(self, inputs_list, targets_list):
		#convert inputs and targets lists to 2d array
		inputs = numpy.array(inputs_list, ndmin=2).T
		targets = numpy.array(targets_list, ndmin=2).T

		# calculate signals into hidden layer
		hidden_inputs = numpy.dot(self.wih, inputs)
		# calculate the signals emerging from hidden layer
		hidden_outputs = self.activation_function(hidden_inputs)

		# calculate signals into final output layer
		final_inputs = numpy.dot (self.who, hidden_outputs)
		# calculate the signals emerging from final output layer
		final_outputs = self.activation_function(final_inputs)
	
		# error is the (target - actual)
		output_errors = targets - final_outputs

		# hidden layer error is the output_errors, split by weights,
		# recombined at hidden nodes
		hidden_errors = numpy.dot(self.who.T, output_errors)

		# update the weights for the links between the hidden and output layers
		self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))

		# update the weights for the links between the input and hidden layers
		self.wih +- self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs)
)
		pass

	# query the neural network
	def query(self, inputs_list):
		# convert inputs list to 2d array
		inputs = numpy.array(inputs_list, ndmin=2).T

		# this combines all inputs with all the right link weights to create
		#  matrix of combined moderated signals into each hidden layer node.
		hidden_inputs = numpy.dot(self.wih, inputs)

		# calculate signals emerging from the hidden layer nodes
		hidden_outputs = self.activation_function(hidden_inputs)

		# calculate signals into final output layer
		final_inputs = numpy.dot(self.who, hidden_outputs)

		#calculate the signals emerging from final output layer
		final_outputs = self.activation_function(final_inputs)

		return final_outputs
		pass


# number of input, hidden, and output nodes
input_nodes = 784
hidden_nodes = 100
output_nodes = 10

# Learning rate is 0.3
learning_rate = 0.3

# create instance of neural network
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# test
# n.query([1.0, 0.5, -1.5])

# Load the mnist training data CSV file into a list
training_data_file = open("MNIST_datasets/training_data/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# train the neural network
# epochs is the number of times the training data set is used for training
epochs = 1

for e in range(epochs):
	# go through all record in the training data set
	for record in training_data_list:
		# split the record by the ',' commas
		all_values = record.split(',')
		# scale and shift the inputs
		inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
		# create the target output values (all 0.01, except the desired label which is 0.99)
		targets = numpy.zeros(output_nodes) + 0.01
		# all_values[0] is the target label for this record
		targets[int(all_values[0])] = 0.99
		n.train(inputs, targets)
		pass
	pass

# load the mnist test data CSV file into a list
test_data_file = open("MNIST_datasets/test_data/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

all_values = test_data_list[0].split(',')
####### Cannot yet display the image, printed array does not always indicate
#######  correct highest value but the conclusion is correct.
# print the label
#print(all_values[0])
#image_array = numpy.asfarray(all_values[1:]).reshape((28,28))
#matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')
#print n.query((numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01)

# test the neural network

# scorecard for how well the network performs, initally empty
scorecard = []

# go through all the record in hte test data set
for record in test_data_list:
	# split the record by the ',' commas
	all_values = record.split(',')
	# correct answer is first value
	correct_label = int(all_values[0])
	#print(correct_label, "correct label")
	# scale and shift the inputs
	inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
	# query the network
	outputs = n.query(inputs)
	# the index of the highest value corresponds to the label
	label = numpy.argmax(outputs)
	#print(label, "network's answer")
	# append correct or incorrect to the list
	if (label == correct_label):
		# network's answer matches correct answer, add 1 to scorecard
		scorecard.append(1)
	else:
		# network's answer does not match correct answer, add 0 to scorecard
		scorecard.append(0)
		pass
# score report!
print "And the network's score is" 
#print(scorecard)
# calculate fraction of correct answers
scorecard_array = numpy.asarray(scorecard)
total = scorecard_array.sum()
size =  scorecard_array.size
print "Percent correct equals %d divided by %d " % (total, size) 
print(10* (10 * total) / size)
# our own image data set
our_own_dataset = []

# Load the png image data as test data set
#for image_file_name in glob.glob('my_own_images/2828_my_own_?.png'):

	
