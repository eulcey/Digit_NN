"""A 3 level Neural Network
to recognise handwritten digits
"""

import math
import numpy as np
from matplotlib import pyplot as plt

TRAINING_IMAGE_FILE = "./data_sets/train-images-idx3-ubyte"
TRAINING_LABEL_FILE = "./data_sets/train-labels-idx1-ubyte"
TEST_IMAGE_FILE = "./data_sets/t10k-images-idx3-ubyte"
TEST_LABEL_FILE = "./data_sets/t10k-labels-idx1-ubyte"

ORIG_INPUTS = 28*28
TRIM_INPUTS = 20*20
TRIMMED = True

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
    try:
        val = math.exp(-x)
    except OverflowError:
        return 0
    else:
        return 1/(1 + math.exp(-x))

def sigmoid_der(x):
    """Returns the value of the derivative of the sigmoid function"""
    return sigmoid(x) * (1 - sigmoid(x))
        

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
    network = NeuralNetwork()
    # Try first to load trained matrices, otherwise generate new weight matrices
    if not network.trainWithFiles():
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
    #print(correct)
    #print(negative)
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
        print("No file given to create_sets")
    image_mat = None
    with open(image_file, "rb") as f:
        magic_number = f.read(4)
        image_count = int.from_bytes(f.read(4), byteorder='big')
        row_count = int.from_bytes(f.read(4), byteorder='big')
        col_count = int.from_bytes(f.read(4), byteorder='big')
        #image_mat = np.zeros([image_count, row_count * col_count]) # old 28*28 image matrix
        image_mat = np.zeros([image_count, TRIM_INPUTS if TRIMMED else ORIG_INPUTS])
        print("Image: %d, row: %d, col: %d" % (image_count, row_count, col_count))
        for i in range(image_count):
            image = list()
            for j in range(row_count):
                # normalization of data should be made already
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
        print("No file given to create_sets")
    label_mat = None
    with open(label_file, "rb") as f:
        magic_number = f.read(4)
        # TODO normalize
        label_count = int.from_bytes(f.read(4), byteorder='big')
        label_mat = np.zeros([label_count])
        print("Labels: %d" % label_count)
        for i in range(label_count):
#            label_list.append(int.from_bytes(f.read(1), byteorder='big'))
            # normalize labels
            label_mat[i] = int.from_bytes(f.read(1), byteorder='big')
    return label_mat



class NeuralNetwork:
    """A 3 level Neural Network"""

    TRAINING_CYCLES = 10


    def __init__(self, inner_neurons=30, function=sigmoid, derivative=sigmoid_der):
        """
        Keyword arguments:
        innerNeurons -- amount of Neurons in the hidden layer
        function -- activation function of the neurons
        """       
        #self.inner_neurons = inner_neurons
        #self.level_one = [[1 for i in range(INPUTS)] for j in range(inner_neurons + 1)]
        self.level_one = np.random.random([inner_neurons, (TRIM_INPUTS if TRIMMED else ORIG_INPUTS) + 1])
        self.level_one = (self.level_one - 0.5) * 2
        #self.level_two = [[1 for i in range(inner_neurons + 1)] for j in range(10)]
        self.level_two = np.random.random([10, inner_neurons])
        self.level_two = (self.level_two - 0.5) * 2
        self.func = np.vectorize(function)
        self.func_der = np.vectorize(derivative)

    def train(self, training_set, training_label):
        """Trains the Network with the given training set"""
        np.save("init_level_one.npy", self.level_one)
        np.save("init_level_two.npy", self.level_two)
        safe = self.level_two
        count = 0
        #a = True
        for i in range(NeuralNetwork.TRAINING_CYCLES):
            for image, label in zip(training_set, training_label):
            #if a:
                count += 1
                if (count % 10000 == 0):
                    print("No on cycle %d" % count)
                #image = training_set[0,:]
                #label = training_label[0]
                #
                target = np.zeros([10])
                target[np.int8(label)] = 1
                data = np.append(image, 1)
                first_step = self.level_one.dot(data)
                test = (self.level_one).dot(data)
                #print(np.max(test))
                first_act_level = self.func(first_step)
                second_step = self.level_two.dot(first_act_level)
                calculated = self.func(second_step)
                delta_out = self.func_der(second_step) * (target - calculated)
                #print(np.shape(delta_out))
                #print(np.shape(first_act_level))
                level_two_change = np.outer(delta_out, first_act_level) # later to add to level_two
                #delta_hidden = self.level_two.dot(self.func_der(first_step)) * delta_out
                # probably the wrong order
                #print(np.shape(delta_hidden))
                delta_hidden = (delta_out.dot(self.level_two)
                                * self.func_der(first_step))
                level_one_change = np.outer(delta_hidden, data)
                #print(np.shape(level_one_change))
                self.level_two = self.level_two + level_two_change
                self.level_one = self.level_one + level_one_change
                #a = False
        np.save("last_level_one.npy", self.level_one)
        np.save("last_level_two.npy", self.level_two)
        #print(self.level_two - safe)

    def trainWithFiles(self):
        """Trains the network with pre-calculated weights, if files are avaible"""
        try:
            self.level_one = np.load(LEVEL_ONE_FILE)
        except FileNotFoundError:
            return False
        try:
            self.level_two = np.load(LEVEL_TWO_FILE)
        except FileNotFoundError:
            return False

    def __calc_prob(self, image):
        """Calculates the network results for each digit on the given image"""
        data = np.append(image, 1) # append bias
        first_step = self.level_one.dot(data)
        first_act_level = self.func(first_step)
        second_step = self.level_two.dot(first_act_level)
        second_act_level = self.func(second_step)
        #print("calc")
        #print(first_act_level)
        #print(second_act_level)
        return second_act_level

    def predict(self, image):
        """Predicts the given data with the trained set"""
        return np.argmax(self.__calc_prob(image)) # only returns first most likely digit for now
