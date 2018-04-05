import pickle
import signal
import sys
import argparse

import tensorflow as tf
from matplotlib import pyplot

from input_function import INPUT_SIZE, OUTPUT_SIZE, create_car_data

NUM_EPOCHS = int(2e4)
BATCH_SIZE = 4
# TODO(velovix): Should this actually be 5 or something else? It's 4 right now
# because my max sequence size is 4
TRUNCATED_BACKPROP_LENGTH = 4
RNN_OUTPUT_SIZE = OUTPUT_SIZE * 5


def on_ctrl_c(signal, frame):
    print("Exiting...")
    sys.exit(0)


def create_model():
    signal.signal(signal.SIGINT, on_ctrl_c)

    argparser = argparse.ArgumentParser(
        description="Attempts to train a network")
    argparser.add_argument("--batch", type=bool,
                           help="Whether or not to use batching")
    argparser.add_argument("--model", type=str,
                           help="Whether or not to use a multilayer RNN")

    args = argparser.parse_args()

    print("Okay! Are we batching?", args.batch,
          "What model are we using?", args.model)

    #car_data = create_car_data()
    #pickle.dump(car_data, open("my_car_data", "wb"))
    car_data = pickle.load(open("my_car_data", "rb"))
    print("Loaded training data: ", len(car_data._input))
    
    input_data, output_data = car_data.training_data()
    print("Input data shape:", input_data.shape)

    if args.batch:
        batch_size = BATCH_SIZE
    else:
        batch_size = len(input_data)
    print("Loaded training data: ", len(input_data))
    print("Using a batch size of: ", batch_size)

    max_time = car_data.max_time()

    input_layer = tf.placeholder(tf.float32,
                                 shape=(batch_size,
                                        TRUNCATED_BACKPROP_LENGTH,
                                        INPUT_SIZE),
                                 name="InputLayer")

    # Create LSTM cells
    if args.model == "big":
        rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in [
            3000, 3000, 3000, 3000, RNN_OUTPUT_SIZE]]
    elif args.model == "small":
        rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in [
            RNN_OUTPUT_SIZE]]
    elif args.model == "medium":
        rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in [
            256, 256, 256, RNN_OUTPUT_SIZE]]

    # create a RNN cell composed sequentially of a number of RNNCells
    multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

    # 'output_layer' is a tensor of shape [batch_size, max_time, 256]
    # 'state' is a N-tuple where N is the number of LSTMCells containing a
    # tf.contrib.rnn.LSTMStateTuple for each cell
    rnn_output_layer, state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                       inputs=input_layer,
                                       dtype=tf.float32)

    output_layer = tf.contrib.layers.fully_connected(
        rnn_output_layer,
        OUTPUT_SIZE)


    print("Created network")

    num_batches = int(len(input_data) / BATCH_SIZE / TRUNCATED_BACKPROP_LENGTH)

    if args.batch:
        print("Training on", num_batches, "batches")

    ground_truth_placeholder = tf.placeholder(tf.float32,
                                              shape=(batch_size,
                                                     TRUNCATED_BACKPROP_LENGTH,
                                                     OUTPUT_SIZE),
                                              name="GroundTruth")

    total_loss = tf.losses.mean_squared_error(ground_truth_placeholder,
                                              output_layer)
    train_step = tf.train.AdamOptimizer().minimize(total_loss)

    pyplot.axis([0, 500, 0, 500])

    loss_count = 0
    highest_loss = 0

    ungraphed_losses = {}

    def train(input_data, output_data):
        _total_loss, _, _output_layer = sess.run(
                [total_loss, train_step, output_layer],
                feed_dict={
                    input_layer: input_data,
                    ground_truth_placeholder: output_data
                })

        return _total_loss

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for epoch_index in range(NUM_EPOCHS):

            if args.batch:
                for batch_index in range(num_batches):
                    start_index = batch_index * BATCH_SIZE
                    end_index = start_index  + BATCH_SIZE

                    batch_input = input_data[start_index:end_index, :]
                    batch_output = output_data[start_index:end_index, :]

                    _total_loss = train(batch_input, batch_output)
            else:
                _total_loss = train(input_data, output_data)

            print(_total_loss)
            if _total_loss > highest_loss:
                highest_loss = _total_loss

            loss_count += 1

            if loss_count % 100 == 0:
                pyplot.axis([0, loss_count, 0, highest_loss])
                pyplot.scatter(loss_count, _total_loss)
                pyplot.pause(0.00001)


    print("done!")
    pyplot.show()


if __name__ == "__main__":
    create_model()
