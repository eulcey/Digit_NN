"""A 3 level Neural Network
to recognise handwritten digits
"""

import numpy as np
from matplotlib import pyplot as plt

TRAINING_IMAGE_FILE = "./data_sets/train-images-idx3-ubyte"
TRAINING_LABEL_FILE = "./data_sets/train-labels-idx1-ubyte"
TEST_IMAGE_FILE = "./data_sets/t10k-images-idx3-ubyte"
TEST_LABEL_FILE = "./data_sets/t10k-labels-idx1-ubyte"

ORIG_INPUTS = 28*28
TRIM_INPUTS = 20*20
TRIMMED = True
TRAIN_NEW = True

TRAIN_MAT_FILE = "train_data.npy" if TRIMMED else "train_data_orig.npy"
TRAINLAB_MAT_FILE = "train_labels.npy" if TRIMMED else "train_labels_orig.npy"
TEST_MAT_FILE = "test_data.npy" if TRIMMED else "test_data_orig.npy"
TESTLAB_MAT_FILE = "test_labels.npy" if TRIMMED else "test_labels_orig.npy"
LEVEL_ONE_FILE = "level_one.npy" if TRIMMED else "level_one_orig.npy"
LEVEL_TWO_FILE = "level_two.npy" if TRIMMED else "level_two_orig.npy"

def show_digit(data, width):
    """Shows a picture of the digit
    data -- NumPy 1D vector of the digit
    width -- width of the digit image
    """
    im = np.reshape(data, (-1, width))
    plt.imshow(im, interpolation='nearest')
    plt.show()

def sigmoid(x):
    """Returns the sigmoid value of x"""
    with np.errstate(over='raise'):
        try:
            val = np.exp(-x)
        except FloatingPointError:
            return 0
        else:
            return 1/(1 + val)

def sigmoid_der(x):
    """Returns the value of the derivative of the sigmoid function"""
    return sigmoid(x) * (1 - sigmoid(x))

def tanh(x):
    """Returns the tanh value of -x"""
    return np.tanh(x)

def tanh_der(x):
    """Returns the value of the derivative of the tanh function"""
    return 1- np.square(np.tanh(x))

INNER_AMOUNT = 30
NN_FUNCTION = sigmoid
NN_DERIVATIVE = sigmoid_der       

def test_network():
    """Tests the prediction error of a neural network"""
    try:
        training_set = np.load(TRAIN_MAT_FILE)
    except FileNotFoundError:
        training_set = decode_images(TRAINING_IMAGE_FILE)
        np.save(TRAIN_MAT_FILE, training_set)
    try:
        training_labels = np.load(TRAINLAB_MAT_FILE)
    except FileNotFoundError:
        training_labels = decode_labels(TRAINING_LABEL_FILE)
        np.save(TRAINLAB_MAT_FILE, training_labels)
    try:
        test_set = np.load(TEST_MAT_FILE)
    except FileNotFoundError:
        test_set = decode_images(TEST_IMAGE_FILE)
        np.save(TEST_MAT_FILE, test_set)
    try:
        test_labels = np.load(TESTLAB_MAT_FILE)
    except FileNotFoundError:
        test_labels = decode_labels(TEST_LABEL_FILE)
        np.save(TESTLAB_MAT_FILE, test_labels)
    network = NeuralNetwork(INNER_AMOUNT, NN_FUNCTION, NN_DERIVATIVE)
    # Try first to load trained matrices, otherwise generate new weight matrices
    if TRAIN_NEW:
        network.train(training_set, training_labels)
    else:
        if not network.trainWithFiles():
            print("Begin training new weights")
            network.train(training_set, training_labels)
    correct = [0 for i in range(10)]
    negative = [0 for i in range(10)]
    amount = [0 for i in range(10)]
    all_correct = 0
    for image, label in zip(test_set,test_labels):
        label = np.int8(label)
        result = network.predict(image)
        if result == label:
            all_correct += 1
            correct[label] += 1
        else:
            negative[result] += 1
        amount[label] += 1
    precision = [true/(true+false) if true > 0 and false > 0 else 0
                 for (true, false) in list(zip(correct, negative))]
    recall = [true/relevant for (true, relevant) in list(zip(correct, amount))]
    print("%f of the test set was correctly classified" % (all_correct/sum(amount)))
    for i in range(10):
        print("For Digit %d precision is %f and recall is %f" % (i, precision[i], recall[i]))
    print("Average precision is %f and recall is %f" % (sum(precision)/10, sum(recall)/10))
    

def decode_images(image_file):
    """Creates a matrix where each row is one image"""
    if image_file is None or image_file is "":
        print("No file given to decode_images")
    image_mat = None
    with open(image_file, "rb") as f:
        magic_number = f.read(4)
        image_count = int.from_bytes(f.read(4), byteorder='big')
        row_count = int.from_bytes(f.read(4), byteorder='big')
        col_count = int.from_bytes(f.read(4), byteorder='big')
        image_mat = np.zeros([image_count, TRIM_INPUTS if TRIMMED else ORIG_INPUTS])
        print("Image: %d, row: %d, col: %d" % (image_count, row_count, col_count))
        for i in range(image_count):
            image = list()
            for j in range(row_count):
                # normalization of data 
                row = [(int.from_bytes(f.read(1),byteorder='big')-127)/127
                       for _ in range(col_count)] 
                image.extend(row)
            # trim row, col 28 -> 20
            image = np.reshape(image, (-1, col_count))
            if TRIMMED:
                image = image[4:-4, 4:-4]
            image_mat[i] = np.array(image.flatten())
    return image_mat 

def decode_labels(label_file):
    """Creates a vector where each value is a label between 0 and 9"""
    if label_file is None or label_file is "":
        print("No file given to decode_labels")
    label_mat = None
    with open(label_file, "rb") as f:
        magic_number = f.read(4)
        # TODO normalize
        label_count = int.from_bytes(f.read(4), byteorder='big')
        label_mat = np.zeros([label_count])
        print("Labels: %d" % label_count)
        for i in range(label_count):
            # normalize labels
            label_mat[i] = int.from_bytes(f.read(1), byteorder='big')
    return label_mat

def plot_error(error_array):
    """Plots the error to the number of cycles"""
    #x = [(i+1) for i in range(NeuralNetwork.TRAINING_CYCLES)]
    #y = list(error_array)
    plt.plot(error_array)
    plt.ylabel('mean square error')
    plt.xlabel('cycle count (in sixty thousend)')
    plt.show()


class NeuralNetwork:
    """A 3 level Neural Network"""

    TRAINING_CYCLES = 5


    def __init__(self, inner_neurons=30, function=sigmoid, derivative=sigmoid_der):
        """
        Keyword arguments:
        innerNeurons -- amount of Neurons in the hidden layer
        function -- activation function of the neurons
        """       
        self.level_one = np.random.random([inner_neurons, (TRIM_INPUTS if TRIMMED else ORIG_INPUTS) + 1])
        self.level_one = (self.level_one - 0.5) * 2
        self.level_two = np.random.random([10, inner_neurons])
        self.level_two = (self.level_two - 0.5) * 2
        self.func = np.vectorize(function)
        self.func_der = np.vectorize(derivative)

    def train(self, training_set, training_label):
        """Trains the Network with the given training set"""
        np.save("init_level_one.npy", self.level_one)
        np.save("init_level_two.npy", self.level_two)
        count = 0
        error = list()
        for i in range(NeuralNetwork.TRAINING_CYCLES):
            cycle_error = 0
            for image, label in zip(training_set, training_label):
                count += 1
                if (count % 10000 == 0):
                    print("Train image cycle: %d" % count)
                target = np.zeros([10])
                target[np.int8(label)] = 1
                data = np.append(image, 1)
                first_step = self.level_one.dot(data)
                test = (self.level_one).dot(data)
                first_act_level = self.func(first_step)
                second_step = self.level_two.dot(first_act_level)
                calculated = self.func(second_step)
                target_error = target - calculated
                cycle_error += np.sqrt(target_error.dot(target_error)/target_error.size)
                delta_out = self.func_der(second_step) * target_error
                level_two_change = np.outer(delta_out, first_act_level) # later to add to level_two
                delta_hidden = (delta_out.dot(self.level_two)
                                * self.func_der(first_step))
                level_one_change = np.outer(delta_hidden, data)
                self.level_two = self.level_two + level_two_change
                self.level_one = self.level_one + level_one_change
            error.append(cycle_error/training_set.size)
        np.save("error_list.npy", np.array(error))
        np.save("last_level_one.npy", self.level_one)
        np.save("last_level_two.npy", self.level_two)

    def trainWithFiles(self):
        """Trains the network with pre-calculated weights, if files are avaible"""
        try:
            self.level_one = np.load(LEVEL_ONE_FILE)
        except FileNotFoundError:
            print("Level One File not found")
            return False
        try:
            self.level_two = np.load(LEVEL_TWO_FILE)
        except FileNotFoundError:
            print("Level Two File not found")
            return False
        return True

    def __calc_prob(self, image):
        """Calculates the network results for each digit on the given image"""
        data = np.append(image, 1) # append bias
        first_step = self.level_one.dot(data)
        first_act_level = self.func(first_step)
        second_step = self.level_two.dot(first_act_level)
        second_act_level = self.func(second_step)
        return second_act_level

    def predict(self, image):
        """Predicts the given data with the trained set"""
        return np.argmax(self.__calc_prob(image)) # only returns first most likely digit for now
