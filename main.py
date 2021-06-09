"""Assignment 4 LayerList for course CS 3B with Eric Reed. This program is the
framework for loading up a dataset and divides it into training and
testing sets. Check-in 3 implemented a weight dictionary for upstream
nodes. Check-in 4 implements a barebones structure for a doubly linked list"""

import unittest
from enum import Enum

import numpy as np
import random
import collections
from abc import ABC, abstractmethod


class DataMismatchError(Exception):
    """Check if there are equal quantities of labels as features in the
    test and training data"""

    def __init__(self, message):
        self.message = message


class LayerType(Enum):
    """Specify elements to be INPUT, HIDDEN, OR OUTPUT"""
    INPUT = 0
    HIDDEN = 1
    OUTPUT = 2


class MultiLinkNode(ABC):
    class Side(Enum):
        UPSTREAM = 0
        DOWNSTREAM = 1

    def __init__(self):
        # Binary encoding to track neighboring nodes when information
        # is available
        self._reporting_nodes = dict.fromkeys([MultiLinkNode.Side.UPSTREAM,
                                               MultiLinkNode.Side.DOWNSTREAM],
                                              0)
        # Represent the reporting node values as binary encoding when
        # all nodes have been reported
        self._reference_value = dict.fromkeys([MultiLinkNode.Side.UPSTREAM,
                                               MultiLinkNode.Side.DOWNSTREAM],
                                              0)
        # References to neighboring nodes upstream and downstream
        self._neighbors = dict.fromkeys([MultiLinkNode.Side.UPSTREAM,
                                         MultiLinkNode.Side.DOWNSTREAM], [])

    def __str__(self):
        all_nodes = list()
        for key, value in self._neighbors.items():
            for v in value:
                all_nodes.append([key, id(v)])
        node_list_upstream = [i[1] for i in all_nodes if i[0] ==
                              MultiLinkNode.Side.UPSTREAM]
        node_list_downstream = [i[1] for i in all_nodes if i[0] ==
                                MultiLinkNode.Side.DOWNSTREAM]
        return f'ID upstream neighboring nodes: {node_list_upstream} \n' \
               f'ID downstream neighboring nodes: {node_list_downstream} \n' \
               f'ID of current node {id(self)}'

    @abstractmethod
    def _process_new_neighbor(self, node, side):
        """Update reference node to be added as a key in the self._weights
        dictionary"""
        pass

    def reset_neighbors(self, nodes: list, side: Side):
        """Resets the nodes that link into this node either upstream or
        downstream. Copies node parameters into appropriate entry of
        neighbors and processes a new neighbor for each node. Then
        calculates and stores the appropriate value in the correct element
        of the reference value list"""
        self._neighbors[side] = nodes.copy()
        for node in nodes:
            self._process_new_neighbor(node, side)
        self._reference_value[side] = (1 << len(nodes)) - 1


class NNData:
    class Order(Enum):
        """Define whether the training data is presented in random or
        sequential order"""
        RANDOM = 0
        SEQUENTIAL = 1

    class Set(Enum):
        """Define if the set is testing data set or the training data
        set"""
        TRAIN = 0
        TEST = 1

    @staticmethod
    def percentage_limiter(percentage: float):
        """Limits training factor percentage between 0-1"""
        if percentage < 0:
            percentage = 0
        if percentage > 1:
            percentage = 1
        return float(percentage)

    def __init__(self, features=None, labels=None, train_factor=0.9):
        self._train_factor = NNData.percentage_limiter(train_factor)
        if features is None:
            features = []
        if labels is None:
            labels = []
        self._features = None
        self._labels = None
        self._train_indices = []
        self._test_indices = []
        self._train_pool = collections.deque()
        self._test_pool = collections.deque()
        try:
            self.load_data(features, labels)
            self.split_set()
        except (ValueError, DataMismatchError):
            pass

    def load_data(self, features, labels):
        """Loads features/labels data into multidimensional array"""
        if features is None or labels is None:
            self._features = None
            self._labels = None
            return
        if len(features) != len(labels):
            raise DataMismatchError("Label and example lists have "
                                    "different lengths")
        try:
            self._features = np.array(features, dtype=float)
            self._labels = np.array(labels, dtype=float)
        except ValueError:
            self._features = None
            self._labels = None
            raise ValueError("Label and example lists must be homogeneous "
                             "and numeric lists of lists")

    def split_set(self, new_train_factor=None):
        """Split training and testing data sets by setting new training
        set factor"""
        if new_train_factor is None:
            new_train_factor = self._train_factor
        else:
            self._train_factor = self.percentage_limiter(new_train_factor)
        sample_size = len(self._features)
        training_size = int(round(sample_size * self._train_factor, 0))
        self._train_indices = random.sample([i for i in range(
            sample_size)], k=training_size)
        self._test_indices = [i for i in range(sample_size) if i not in
                              self._train_indices]

    def prime_data(self, target_set=None, order=None):
        """Copy train/test indices into a train/test pool and order them
        sequentially or randomly"""
        if target_set is None:
            self._train_pool = self._train_indices.copy()
            self._test_pool = self._test_indices.copy()
        elif target_set == NNData.Set.TEST:
            self._test_pool = self._test_indices.copy()
        elif target_set == NNData.Set.TRAIN:
            self._train_pool = self._train_indices.copy()
        if order == NNData.Order.RANDOM:
            print(f"Random order instigated")
            random.shuffle(self._train_pool)
            random.shuffle(self._test_pool)
        self._train_pool = collections.deque(self._train_pool)
        self._test_pool = collections.deque(self._test_pool)

    def get_one_item(self, target_set=None):
        """Gather one pair of train and test values"""
        if target_set == NNData.Set.TRAIN or target_set is None:
            index = self._train_pool.popleft()
            feature_value = self._features[index]
            label_value = self._labels[index]
        elif target_set == NNData.Set.TEST:
            print(f"Target set successfully changed to TEST")
            index = self._test_pool.popleft()
            feature_value = self._features[index]
            label_value = self._labels[index]
        return (feature_value, label_value)

    def number_of_samples(self, target_set=None):
        """Calculate the sample size of the specified set (train/test)"""
        if target_set == NNData.Set.TRAIN:
            target_set_sample_size = len(self._features)
        elif target_set == NNData.Set.TEST:
            target_set_sample_size = len(self._features)
        else:
            target_set_sample_size = len(self._features)
        return target_set_sample_size

    def pool_is_empty(self, target_set=None):
        """Return false if the target set (test/train) is empty"""
        empty_set = True
        if target_set == NNData.Set.TRAIN or target_set is None:
            target_set = self._train_pool
        elif target_set == NNData.Set.TEST:
            target_set = self._test_pool
        if bool(target_set):
            empty_set = False
        return empty_set


class Neurode(MultiLinkNode):

    def __init__(self, node_type, learning_rate=.05):
        super().__init__()
        self._value = 0
        self._node_type = node_type
        self._learning_rate = learning_rate
        self._weights = {}

    def _process_new_neighbor(self, node, side: MultiLinkNode.Side):
        """Updates weight dictionary if UPSTREAM node is generated with a
        randomly generated float between 0 and 1 when a new node is added"""
        if side is MultiLinkNode.Side.UPSTREAM:
            self._weights[node] = random.random()

    def _check_in(self, node, side: MultiLinkNode.Side):
        """Check in upstream nodes that have information to update the
        following node with"""
        node_number = self._neighbors[side].index(node)
        self._reporting_nodes[side] = \
            self._reporting_nodes[side] | 1 << node_number
        if self._reporting_nodes[side] == self._reference_value[side]:
            self._reporting_nodes[side] = 0
            return True
        else:
            return False

    @property
    def value(self):
        return self._value

    @property
    def node_type(self):
        return self._node_type

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate: float):
        self._learning_rate = learning_rate

    def get_weight(self, node):
        """Get weights of previously passed upstream nodes relative to
        current node"""
        return self._weights[node]


class FFNeurode(Neurode):

    def __init__(self, my_type):
        super().__init__(my_type)

    @staticmethod
    def _sigmoid(value):
        """Result of sigmoid function at value"""
        return 1 / (1 + np.exp(-value))

    def _calculate_value(self):
        """Calculate the weighted sum of the upstream nodes' values and
        pass through the sigmoid function"""
        input_sum = 0
        for node, weight in self._weights.items():
            input_sum += node.value * weight
        self._value = self._sigmoid(input_sum)

    def _fire_downstream(self):
        """Send values upstream"""
        for node in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            node.data_ready_upstream(self)

    def data_ready_upstream(self, from_node):
        """Check if node has upstream node data ready to send to the
        downstream node"""
        if self._check_in(from_node, MultiLinkNode.Side.UPSTREAM):
            self._calculate_value()
            self._fire_downstream()

    def set_input(self, input_value: float):
        """Set value of input layer node"""
        self._value = input_value
        for node in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            node.data_ready_upstream(self)


class BPNeurode(Neurode):

    def __init__(self, my_type):
        super().__init__(my_type)
        self._delta = 0

    @property
    def delta(self):
        return self._delta

    @staticmethod
    def _sigmoid_derivative(value):
        return value * (1.0 - value)

    def _calculate_delta(self, expected_value=None):
        """Calculate the delta for output, hidden, input nodes"""
        if self._node_type == LayerType.OUTPUT:
            error = expected_value - self.value
            self._delta = error * self._sigmoid_derivative(self.value)
        else:
            self._delta = 0
            for neurode in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
                self._delta += neurode.get_weight(self) * neurode.delta
            self._delta *= self._sigmoid_derivative(self.value)

    def set_expected(self, expected_value: float):
        self._calculate_delta(expected_value)
        self._fire_upstream()

    def data_ready_downstream(self, from_node):
        """Check if data is ready downstream and collect the data to
        make it available in the next layer up after calculating the
        current node's delta value, and update the weights"""
        if self._check_in(from_node, MultiLinkNode.Side.DOWNSTREAM):
            self._calculate_delta()
            self._fire_upstream()
            self._update_weights()

    def adjust_weights(self, node, adjustment):
        """Set strength of node importance by using node reference to
        add adjustments to appropriate entry of weights """
        self._weights[node] += adjustment

    def _update_weights(self):
        """Iterate through downstream neighbors and use adjust_weights
        method to request an adjustment to weight/significance of
        current node's data"""
        for node in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            adjustment = node.learning_rate * node.delta * self.value
            node.adjust_weights(self, adjustment)

    def _fire_upstream(self):
        for node in self._neighbors[MultiLinkNode.Side.UPSTREAM]:
            node.data_ready_downstream(self)


class FFBPNeurode(FFNeurode, BPNeurode):
    pass


class Node:
    def __init__(self, data=None):
        self.data = data
        self.next = None
        self.prev = None


class DoublyLinkedList:
    """Class makes current the head node when first item is added."""

    class EmptyListError(Exception):
        """Check if list is empty and return custom error message if list
        is empty"""

        def __init__(self, message):
            self.message = message

    def __init__(self):
        self._head = None
        self._curr = None
        self._tail = None

    def __iter__(self):
        return self

    def __next__(self):
        if self._curr and self._curr.next:
            ret_val = self._curr.data
            self._curr = self._curr.next
            return ret_val
        raise StopIteration

    def move_forward(self):
        """Check if current node is empty or the tail node. Set new current
        node to be the next one in the doublylinkedlist"""
        if not self._curr:
            raise DoublyLinkedList.EmptyListError("Empty list - cannot "
                                                  "move to next object")
        # If there is another node after the current node, set the next
        # node to be the current node, otherwise the current node is the tail
        # and raise an Index Error
        if self._curr.next:
            self._curr = self._curr.next
        else:
            raise IndexError

    def move_back(self):
        """Check if the current node is empty or the tail node. If neither,
        then set the new current node to the next node in the doubly linked
        list"""
        if not self._curr:
            raise DoublyLinkedList.EmptyListError("List empty - cannot move "
                                                  "back")
        if self._curr.prev:
            self._curr = self._curr.prev
        else:
            raise IndexError

    def reset_to_head(self):
        """Set the current node to be the head node"""
        if not self._head:
            raise DoublyLinkedList.EmptyListError("Empty list - cannot "
                                                  "reset to head")
        self._curr = self._head

    def reset_to_tail(self):
        """Set current node to be the tail node"""
        if not self._tail:
            raise DoublyLinkedList.EmptyListError("Empty list")
        self._curr = self._tail

    def add_to_head(self, data):
        """Add a new node to the doubly linked list at the head node
        position if there already exists a head. If there is only one node,
        set the head equal to the tail"""
        new_node = Node(data)
        new_node.next = self._head
        if self._head:
            self._head.prev = new_node
        self._head = new_node
        if self._tail is None:
            self._tail = new_node
        self.reset_to_head()

    def add_after_cur(self, data):
        """Add a new node after the current node."""
        if not self._curr:
            raise self.EmptyListError
        new_node = Node(data)
        new_node.prev = self._curr
        new_node.next = self._curr.next
        if self._curr.next:
            self._curr.next.prev = new_node
        self._curr.next = new_node
        if self._tail == self._curr:
            self._tail = new_node

    def remove_from_head(self):
        """Remove the head node in the doubly linked list"""
        if not self._head:
            raise DoublyLinkedList.EmptyListError("Nothing to remove from "
                                                  "head of empty list")
        ret_val = self._head.data
        self._head = self._head.next
        if self._head:
            self._head.prev = None
        else:
            self._tail = None
        self.reset_to_head()
        return ret_val

    def remove_after_cur(self):
        """Remove the node that is directly after the current node"""
        if not self._curr:
            raise DoublyLinkedList.EmptyListError("No data to remove in "
                                                  "remove_after_cur")
        # Cannot delete node after tail node
        if self._curr.next is None:
            raise IndexError
        ret_val = self._curr.next.data
        # If current node is second to last, delete last node
        if self._curr.next == self._tail:
            self._tail = self._curr
            self._curr.next = None
            print("Deleting last node")
        # Delete node after current node
        else:
            self._curr.next = self._curr.next.next
            self._curr.next.prev = self._curr
        return ret_val

    def get_current_data(self):
        """View the current node's data"""
        if not self._curr:
            raise DoublyLinkedList.EmptyListError("Dataset Empty in "
                                                  "get_current_data")
        return self._curr.data


class LayerList(DoublyLinkedList, FFBPNeurode):
    def __init__(self, inputs: int, outputs: int):
        super().__init__()
        input_nodes = []
        output_nodes = []
        # Randomly generate input and output FFBPNeurode objects according
        # to user input/output values
        for _ in range(inputs):
            input_nodes.append(FFBPNeurode(LayerType.INPUT))
        for _ in range(outputs):
            output_nodes.append(FFBPNeurode(LayerType.OUTPUT))
        # Link the input and output layers together
        for node in input_nodes:
            node.reset_neighbors(output_nodes,
                                 MultiLinkNode.Side.DOWNSTREAM)
        for node in output_nodes:
            node.reset_neighbors(input_nodes, MultiLinkNode.Side.UPSTREAM)
        # Add FFBPNeurode objects into the doubly linked list
        self._input_nodes = input_nodes
        self._output_nodes = output_nodes
        self.add_to_head(self._input_nodes)
        self.add_after_cur(self._output_nodes)
        self._num_layers = 2

    def add_layer(self, num_nodes: int):
        """Add hidden layer stored in a node after the current layer"""
        if self._curr == self._tail:
            raise IndexError
        hidden_nodes = []
        for _ in range(num_nodes):
            hidden_nodes.append(FFBPNeurode(LayerType.HIDDEN))
        # There is only the head and tail nodes in the DoublyLinkedList
        if self._num_layers == 2:
            for neurode in hidden_nodes:
                neurode.reset_neighbors(self._input_nodes,
                                        MultiLinkNode.Side.UPSTREAM)
                neurode.reset_neighbors(self._output_nodes,
                                        MultiLinkNode.Side.DOWNSTREAM)
            self.add_after_cur(hidden_nodes)
        else:
            for neurode in hidden_nodes:
                neurode.reset_neighbors(self.get_current_data(),
                                        MultiLinkNode.Side.UPSTREAM)
            self.add_after_cur(hidden_nodes)
            self.move_forward()
            for neurode in hidden_nodes:
                neurode.reset_neighbors(self.get_current_data(),
                                        MultiLinkNode.Side.DOWNSTREAM)
            self.move_back()
        self._num_layers += 1

    def remove_layer(self):
        """Remove the layer AFTER the current layer in the linked list node
        - but do not let client remove output layer"""
        if self._curr.next == self._tail:
            raise IndexError
        else:
            self.remove_after_cur()
            self._num_layers -= 1

    @property
    def input_nodes(self):
        return self._input_nodes

    @property
    def output_nodes(self):
        return self._output_nodes


def load_XOR():
    """Load mock XOR data for features and labels sets"""
    features = [[0, 0], [1, 0], [0, 1], [1, 1]]
    labels = [[0], [1], [1], [0]]
    training_factor = 1
    testing_features_and_labels = NNData()
    testing_features_and_labels.load_data(features, labels)


def unit_test():
    """Compilation of unit tests from Prof Eric Reed"""
    errors = False
    try:
        # Create a valid small and large dataset to be used later
        x = [[i] for i in range(10)]
        y = x
        our_data_0 = NNData(x, y)
        x = [[i] for i in range(100)]
        y = x
        our_big_data = NNData(x, y, .5)

        # Try loading lists of different sizes
        y = [[1]]
        try:
            our_bad_data = NNData()
            our_bad_data.load_data(x, y)
            raise Exception
        except DataMismatchError:
            print("DataMismatchError raised - PASSED properly")
            pass
        except:
            raise Exception
        # Create a dataset that can be used to make sure the
        # features and labels are not confused
        x = [[1.0], [2.0], [3.0], [4.0]]
        y = [[.1], [.2], [.3], [.4]]
        our_data_1 = NNData(x, y, .5)
    except:
        print("There are errors that likely come from __init__ or a "
              "method called by __init__")
        errors = True

    # Test split_set to make sure the correct number of examples are in
    # each set, and that the indices do not overlap.
    try:
        our_data_0.split_set(.3)
        print(f"Train Indices:{our_data_0._train_indices}")
        print(f"Test Indices:{our_data_0._test_indices}")

        assert len(our_data_0._train_indices) == 3
        assert len(our_data_0._test_indices) == 7
        assert (list(set(our_data_0._train_indices +
                         our_data_0._test_indices))) == list(range(10))
    except:
        print("There are errors that likely come from split_set")
        errors = True
    if errors == False:
        print("TESTING - split_method() PASSED \n")
    # Make sure prime_data sets up the deques correctly, whether
    # sequential or random.
    try:
        our_data_0.prime_data(order=NNData.Order.SEQUENTIAL)
        assert len(our_data_0._train_pool) == 3
        assert len(our_data_0._test_pool) == 7
        assert our_data_0._train_indices == list(our_data_0._train_pool)
        assert our_data_0._test_indices == list(our_data_0._test_pool)
        our_big_data.prime_data(order=NNData.Order.RANDOM)
        assert our_big_data._train_indices != list(our_big_data._train_pool)
        assert our_big_data._test_indices != list(our_big_data._test_pool)
    except:
        print("There are errors that likely come from prime_data")
        errors = True

    # Make sure get_one_item is returning the correct values, and
    # that pool_is_empty functions correctly.
    try:
        our_data_1.prime_data(order=NNData.Order.SEQUENTIAL)
        my_x_list = []
        my_y_list = []
        while not our_data_1.pool_is_empty():
            example = our_data_1.get_one_item()
            my_x_list.append(list(example[0]))
            my_y_list.append(list(example[1]))
        assert len(my_x_list) == 2
        assert my_x_list != my_y_list
        my_matched_x_list = [i[0] * 10 for i in my_y_list]
        print(my_matched_x_list)
        assert my_matched_x_list == my_x_list
        while not our_data_1.pool_is_empty(our_data_1.Set.TEST):
            example = our_data_1.get_one_item(our_data_1.Set.TEST)
            my_x_list.append(list(example[0]))
            my_y_list.append(list(example[1]))
        assert my_x_list != my_y_list
        my_matched_x_list = [i[0] * 10 for i in my_y_list]
        assert my_matched_x_list == my_x_list
        assert set(i[0] for i in my_x_list) == set(i[0] for i in x)
        assert set(i[0] for i in my_y_list) == set(i[0] for i in y)
    except:
        print("There are errors that may come from prime_data, but could "
              "be from another method")
        errors = True

    # Summary
    if errors:
        print("You have one or more errors.  Please fix them before "
              "submitting")
    else:
        print("No errors were identified by the unit test.")
        print("You should still double check that your code meets spec.")
        print("You should also check that PyCharm does not identify any "
              "PEP-8 issues.")


def check_point_one_test():
    # Mock up a network with three inputs and three outputs

    inputs = [Neurode(LayerType.INPUT) for _ in range(3)]
    outputs = [Neurode(LayerType.OUTPUT, .01) for _ in range(3)]
    if not inputs[0]._reference_value[MultiLinkNode.Side.DOWNSTREAM] == 0:
        print("Fail - Initial reference value is not zero")
    for node in inputs:
        node.reset_neighbors(outputs, MultiLinkNode.Side.DOWNSTREAM)
    for node in outputs:
        node.reset_neighbors(inputs, MultiLinkNode.Side.UPSTREAM)
    if not inputs[0]._reference_value[MultiLinkNode.Side.DOWNSTREAM] == 7:
        print("Fail - Final reference value is not correct")
    if not inputs[0]._reference_value[MultiLinkNode.Side.UPSTREAM] == 0:
        print("Fail - Final reference value is not correct")

    # Report data ready from each input and make sure _check_in
    # only reports True when all nodes have reported

    if not outputs[0]._reporting_nodes[MultiLinkNode.Side.UPSTREAM] == 0:
        print("Fail - Initial reporting value is not zero")
    if outputs[0]._check_in(inputs[0], MultiLinkNode.Side.UPSTREAM):
        print("Fail - _check_in returned True but not all nodes were"
              "checked in")
    if not outputs[0]._reporting_nodes[MultiLinkNode.Side.UPSTREAM] == 1:
        print("Fail - 1st reporting value is not correct")
    if outputs[0]._check_in(inputs[2], MultiLinkNode.Side.UPSTREAM):
        print("Fail - _check_in returned True but not all nodes were"
              "checked in")
    if not outputs[0]._reporting_nodes[MultiLinkNode.Side.UPSTREAM] == 5:
        print("Fail - 2nd reporting value is not correct")
    if outputs[0]._check_in(inputs[2], MultiLinkNode.Side.UPSTREAM):
        print("Fail - _check_in returned True but not all nodes were"
              "checked in (double fire)")
    if not outputs[0]._reporting_nodes[MultiLinkNode.Side.UPSTREAM] == 5:
        print("Fail - 3rd reporting value is not correct")
    if not outputs[0]._check_in(inputs[1], MultiLinkNode.Side.UPSTREAM):
        print("Fail - _check_in returned False after all nodes were"
              "checked in")

    # Report data ready from each output and make sure _check_in
    # only reports True when all nodes have reported

    if inputs[1]._check_in(outputs[0], MultiLinkNode.Side.DOWNSTREAM):
        print("Fail - _check_in returned True but not all nodes were"
              "checked in")
    if inputs[1]._check_in(outputs[2], MultiLinkNode.Side.DOWNSTREAM):
        print("Fail - _check_in returned True but not all nodes were"
              "checked in")
    if inputs[1]._check_in(outputs[0], MultiLinkNode.Side.DOWNSTREAM):
        print("Fail - _check_in returned True but not all nodes were"
              "checked in (double fire)")
    if not inputs[1]._check_in(outputs[1], MultiLinkNode.Side.DOWNSTREAM):
        print("Fail - _check_in returned False after all nodes were"
              "checked in")
    #
    # # Check that learning rates were set correctly

    if not inputs[0].learning_rate == .05:
        print("Fail - default learning rate was not set")
    if not outputs[0].learning_rate == .01:
        print("Fail - specified learning rate was not set")

    # Check that weights appear random

    weight_list = list()
    for node in outputs:
        print(node.__str__())
        for t_node in inputs:
            if node.get_weight(t_node) in weight_list:
                print("Fail - weights do not appear to be set up properly")
            weight_list.append(node.get_weight(t_node))


def check_point_two_test():
    inodes = []
    hnodes = []
    onodes = []
    for k in range(2):
        inodes.append(FFNeurode(LayerType.INPUT))
    for k in range(2):
        hnodes.append(FFNeurode(LayerType.HIDDEN))
    onodes.append(FFNeurode(LayerType.OUTPUT))
    for node in inodes:
        node.reset_neighbors(hnodes, MultiLinkNode.Side.DOWNSTREAM)
    for node in hnodes:
        node.reset_neighbors(inodes, MultiLinkNode.Side.UPSTREAM)
        node.reset_neighbors(onodes, MultiLinkNode.Side.DOWNSTREAM)
    for node in onodes:
        node.reset_neighbors(hnodes, MultiLinkNode.Side.UPSTREAM)
    try:
        inodes[1].set_input(1)
        assert onodes[0].value == 0
    except:
        print("Error: Neurodes may be firing before receiving all input")
    inodes[0].set_input(0)

    # Since input node 0 has value of 0 and input node 1 has value of
    # one, the value of the hidden layers should be the sigmoid of the
    # weight out of input node 1.

    value_0 = (1 / (1 + np.exp(-hnodes[0]._weights[inodes[1]])))
    value_1 = (1 / (1 + np.exp(-hnodes[1]._weights[inodes[1]])))
    inter = onodes[0]._weights[hnodes[0]] * value_0 + \
            onodes[0]._weights[hnodes[1]] * value_1
    final = (1 / (1 + np.exp(-inter)))
    print(value_0, value_1, inter, final, onodes[0].value)
    try:
        print(final, onodes[0].value)
        assert final == onodes[0].value
        assert 0 < final < 1
    except:
        print("Error: Calculation of neurode value may be incorrect")


class TestNNData(unittest.TestCase):
    """Unit test for NNData class and unit test examples"""

    def test_custom_errors(self):
        """Test if custom errors are raised"""
        test_NNData = NNData()
        with self.assertRaises(ValueError):
            # Tests if features or labels contain non-float values when
            # calling NNData.load_data()
            test_NNData.load_data(['a'], [1])
        with self.assertRaises(DataMismatchError):
            # Test if function NNData.load_data() raises the custom
            # exception DataMismatchError if features and labels have
            # different lengths when calling.
            test_NNData.load_data([1], [1, 2])
        with self.assertRaises(TypeError):
            # Verify that invalid data values sets features/labels to None
            test_NNData.load_data(1)
            self.assertIsNone(test_NNData.features)

    def test_training_factor_limit(self):
        """Test if NNData class properly limits training factor"""
        test_NNData = NNData()
        # Verify that percentage_limiter limits negative numbers to 0
        self.assertEqual(0, test_NNData.percentage_limiter(-1))
        # Verify that percentage_limiter limits numbers greater than 1
        # to 1
        self.assertEqual(1, test_NNData.percentage_limiter(5))


def main():
    try:
        test_neurode = BPNeurode(LayerType.HIDDEN)
    except:
        print("Error - Cannot instaniate a BPNeurode object")
        return
    print("Testing Sigmoid Derivative")
    try:
        assert BPNeurode._sigmoid_derivative(0) == 0
        if test_neurode._sigmoid_derivative(.4) == .24:
            print("Pass")
        else:
            print("_sigmoid_derivative is not returning the correct "
                  "result")
    except:
        print("Error - Is _sigmoid_derivative named correctly, created "
              "in BPNeurode and decorated as a static method?")
    print("Testing Instance objects")
    try:
        test_neurode.learning_rate
        test_neurode.delta
        print("Pass")
    except:
        print("Error - Are all instance objects created in __init__()?")

    inodes = []
    hnodes = []
    onodes = []
    for k in range(2):
        inodes.append(FFBPNeurode(LayerType.INPUT))
        hnodes.append(FFBPNeurode(LayerType.HIDDEN))
        onodes.append(FFBPNeurode(LayerType.OUTPUT))
    for node in inodes:
        node.reset_neighbors(hnodes, MultiLinkNode.Side.DOWNSTREAM)
    for node in hnodes:
        node.reset_neighbors(inodes, MultiLinkNode.Side.UPSTREAM)
        node.reset_neighbors(onodes, MultiLinkNode.Side.DOWNSTREAM)
    for node in onodes:
        node.reset_neighbors(hnodes, MultiLinkNode.Side.UPSTREAM)
    print("testing learning rate values")
    for node in hnodes:
        print(f"my learning rate is {node.learning_rate}")
    print("Testing check-in")
    try:
        hnodes[0]._reporting_nodes[MultiLinkNode.Side.DOWNSTREAM] = 1
        if hnodes[0]._check_in(onodes[1], MultiLinkNode.Side.DOWNSTREAM) and \
                not hnodes[1]._check_in(onodes[1],
                                        MultiLinkNode.Side.DOWNSTREAM):
            print("Pass")
        else:
            print("Error - _check_in is not responding correctly")
    except:
        print("Error - _check_in is raising an error.  Is it named correctly? "
              "Check your syntax")
    print("Testing calculate_delta on output nodes")
    try:
        onodes[0]._value = .2
        onodes[0]._calculate_delta(.5)
        if .0479 < onodes[0].delta < .0481:
            print("Pass")
        else:
            print("Error - calculate delta is not returning the correct value."
                  "Check the math.")
            print("        Hint: do you have a separate process for hidden "
                  "nodes vs output nodes?")
    except:
        print("Error - calculate_delta is raising an error.  Is it named "
              "correctly?  Check your syntax")
    print("Testing calculate_delta on hidden nodes")
    try:
        onodes[0]._delta = .2
        onodes[1]._delta = .1
        onodes[0]._weights[hnodes[0]] = .4
        onodes[1]._weights[hnodes[0]] = .6
        hnodes[0]._value = .3
        hnodes[0]._calculate_delta()
        if .02939 < hnodes[0].delta < .02941:
            print("Pass")
        else:
            print(
                "Error - calculate delta is not returning the correct value.  "
                "Check the math.")
            print("        Hint: do you have a separate process for hidden "
                  "nodes vs output nodes?")
    except:
        print(
            "Error - calculate_delta is raising an error.  Is it named correctly?  Check your syntax")
    try:
        print("Testing update_weights")
        hnodes[0]._update_weights()
        if onodes[0].learning_rate == .05:
            if .4 + .06 * onodes[0].learning_rate - .001 < \
                    onodes[0]._weights[hnodes[0]] < \
                    .4 + .06 * onodes[0].learning_rate + .001:
                print("Pass")
            else:
                print("Error - weights not updated correctly.  "
                      "If all other methods passed, check update_weights")
        else:
            print("Error - Learning rate should be .05, please verify")
    except:
        print("Error - update_weights is raising an error.  Is it named "
              "correctly?  Check your syntax")
    print("All that looks good.  Trying to train a trivial dataset "
          "on our network")
    inodes = []
    hnodes = []
    onodes = []
    for k in range(2):
        inodes.append(FFBPNeurode(LayerType.INPUT))
        hnodes.append(FFBPNeurode(LayerType.HIDDEN))
        onodes.append(FFBPNeurode(LayerType.OUTPUT))
    for node in inodes:
        node.reset_neighbors(hnodes, MultiLinkNode.Side.DOWNSTREAM)
    for node in hnodes:
        node.reset_neighbors(inodes, MultiLinkNode.Side.UPSTREAM)
        node.reset_neighbors(onodes, MultiLinkNode.Side.DOWNSTREAM)
    for node in onodes:
        node.reset_neighbors(hnodes, MultiLinkNode.Side.UPSTREAM)
    inodes[0].set_input(1)
    inodes[1].set_input(0)
    value1 = onodes[0].value
    value2 = onodes[1].value
    onodes[0].set_expected(0)
    onodes[1].set_expected(1)
    inodes[0].set_input(1)
    inodes[1].set_input(0)
    value1a = onodes[0].value
    value2a = onodes[1].value
    if (value1 - value1a > 0) and (value2a - value2 > 0):
        print("Pass - Learning was done!")
    else:
        print("Fail - the network did not make progress.")
        print("If you hit a wall, be sure to seek help in the discussion "
              "forum, from the instructor and from the tutors")


def dll_test():
    my_list = DoublyLinkedList()
    try:
        my_list.get_current_data()
    except DoublyLinkedList.EmptyListError:
        print("Pass - EmptyListError properly raised")
    else:
        print("Fail - EmptyListError is not raised")
    for a in range(3):
        my_list.add_to_head(a)
    if my_list.get_current_data() != 2:
        print("Error")
    my_list.move_forward()
    if my_list.get_current_data() != 1:
        print("Fail - get_current_data method")
    my_list.move_forward()
    try:
        my_list.move_forward()
    except IndexError:
        print("Pass - Moving Forward method IndexError properly raised")
    else:
        print("Fail - move_forward method")
    if my_list.get_current_data() != 0:
        print("Fail - current node not at tail")
    my_list.move_back()
    my_list.remove_after_cur()
    if my_list.get_current_data() != 1:
        print("Fail - remove_After_cur did not set new tail")
    my_list.move_back()
    if my_list.get_current_data() != 2:
        print("Fail")
    try:
        my_list.move_back()
    except IndexError:
        print("Pass - Moving backwards IndexError properly raised")
    else:
        print("Fail")
    my_list.move_forward()
    if my_list.get_current_data() != 1:
        print("Fail")


def layer_list_test():
    # create a LayerList with two inputs and four outputs
    my_list = LayerList(2, 4)
    # get a list of the input and output nodes, and make sure we have the right number
    inputs = my_list.input_nodes
    outputs = my_list.output_nodes
    assert len(inputs) == 2
    assert len(outputs) == 4
    # check that each has the right number of connections
    for node in inputs:
        assert len(node._neighbors[MultiLinkNode.Side.DOWNSTREAM]) == 4
    for node in outputs:
        assert len(node._neighbors[MultiLinkNode.Side.UPSTREAM]) == 2
    # check that the connections go to the right place
    for node in inputs:
        out_set = set(node._neighbors[MultiLinkNode.Side.DOWNSTREAM])
        check_set = set(outputs)
        assert out_set == check_set
    for node in outputs:
        in_set = set(node._neighbors[MultiLinkNode.Side.UPSTREAM])
        check_set = set(inputs)
        assert in_set == check_set
    # add a couple layers and check that they arrived in the right order,
    # and that iterate and rev_iterate work
    my_list.reset_to_head()
    my_list.add_layer(3)
    my_list.add_layer(6)
    my_list.move_forward()
    assert my_list.get_current_data()[0].node_type == LayerType.HIDDEN
    assert len(my_list.get_current_data()) == 6
    my_list.move_forward()
    assert my_list.get_current_data()[0].node_type == LayerType.HIDDEN
    assert len(my_list.get_current_data()) == 3
    # save this layer to make sure it gets properly removed later
    my_list.move_forward()
    assert my_list.get_current_data()[0].node_type == LayerType.OUTPUT
    assert len(my_list.get_current_data()) == 4
    my_list.move_back()
    assert my_list.get_current_data()[0].node_type == LayerType.HIDDEN
    assert len(my_list.get_current_data()) == 3
    # check that information flows through all layers
    save_vals = []
    for node in outputs:
        save_vals.append(node.value)
    for node in inputs:
        node.set_input(1)
    for i, node in enumerate(outputs):
        assert save_vals[i] != node.value
    # check that information flows back as well
    save_vals = []
    for node in inputs[1]._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
        save_vals.append(node.delta)
    for node in outputs:
        node.set_expected(1)
    for i, node in enumerate(
            inputs[1]._neighbors[MultiLinkNode.Side.DOWNSTREAM]):
        assert save_vals[i] != node.delta
    # try to remove an output layer
    try:
        print("Trying to remove an output layer...")
        my_list.remove_layer()
        assert False
    except IndexError:
        print("IndexError properly raised")
        pass
    except:
        assert False
    # move and remove a hidden layer
    save_list = my_list.get_current_data()
    my_list.move_back()
    my_list.remove_layer()
    # check the order of layers again
    my_list.reset_to_head()
    assert my_list.get_current_data()[0].node_type == LayerType.INPUT
    assert len(my_list.get_current_data()) == 2
    my_list.move_forward()
    assert my_list.get_current_data()[0].node_type == LayerType.HIDDEN
    assert len(my_list.get_current_data()) == 6
    my_list.move_forward()
    assert my_list.get_current_data()[0].node_type == LayerType.OUTPUT
    assert len(my_list.get_current_data()) == 4
    my_list.move_back()
    assert my_list.get_current_data()[0].node_type == LayerType.HIDDEN
    assert len(my_list.get_current_data()) == 6
    my_list.move_back()
    assert my_list.get_current_data()[0].node_type == LayerType.INPUT
    assert len(my_list.get_current_data()) == 2
    # save a value from the removed layer to make sure it doesn't get changed
    saved_val = save_list[0].value
    # check that information still flows through all layers
    save_vals = []
    for node in outputs:
        save_vals.append(node.value)
    for node in inputs:
        node.set_input(1)
    for i, node in enumerate(outputs):
        assert save_vals[i] != node.value
    # check that information still flows back as well
    save_vals = []
    for node in inputs[1]._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
        save_vals.append(node.delta)
    for node in outputs:
        node.set_expected(1)
    for i, node in enumerate(
            inputs[1]._neighbors[MultiLinkNode.Side.DOWNSTREAM]):
        assert save_vals[i] != node.delta
    assert saved_val == save_list[0].value


if __name__ == "__main__":
    layer_list_test()
