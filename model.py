import tensorflow as tf
import pickle
from input_function import INPUT_SIZE, OUTPUT_SIZE, create_car_data

NUM_EPOCHS = 100
BATCH_SIZE = 5
# TODO(velovix): Should this actually be 5 or something else? It's 4 right now
# because my max sequence size is 4
TRUNCATED_BACKPROP_LENGTH = 4


def create_model():
    car_data = create_car_data()
    pickle.dump(car_data, open("my_car_data", "wb"))
    #car_data = pickle.load(open("my_car_data", "rb"))
    print("Loaded training data")

    max_time = car_data.max_time()

    input_layer = tf.placeholder(tf.float32,
                                 shape=(BATCH_SIZE,
                                        TRUNCATED_BACKPROP_LENGTH,
                                        INPUT_SIZE),
                                 name="InputLayer")

    # create 2 LSTMCells
    rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in [128, 5]]

    # create a RNN cell composed sequentially of a number of RNNCells
    multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

    # 'output_layer' is a tensor of shape [batch_size, max_time, 256]
    # 'state' is a N-tuple where N is the number of LSTMCells containing a
    # tf.contrib.rnn.LSTMStateTuple for each cell
    output_layer, state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                       inputs=input_layer,
                                       dtype=tf.float32)

    print("Created network")

    input_data, output_data = car_data.training_data()
    print("Input data shape:", input_data.shape)

    num_batches = int(len(input_data) / BATCH_SIZE / TRUNCATED_BACKPROP_LENGTH)

    print("Training on", num_batches, "batches")

    ground_truth_placeholder = tf.placeholder(tf.float32,
                                              shape=(BATCH_SIZE,
                                                     TRUNCATED_BACKPROP_LENGTH,
                                                     OUTPUT_SIZE),
                                              name="GroundTruth")

    total_loss = tf.reduce_sum(tf.abs(output_layer - ground_truth_placeholder))
    train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for epoch_index in range(NUM_EPOCHS):

            for batch_index in range(num_batches):
                start_index = batch_index * BATCH_SIZE
                end_index = start_index  + BATCH_SIZE

                batch_input = input_data[start_index:end_index, :]
                batch_output = output_data[start_index:end_index, :]

                _total_loss, _,  = sess.run(
                        [total_loss, train_step],
                        feed_dict={
                            input_layer: batch_input,
                            output_layer: batch_output,
                            ground_truth_placeholder: batch_output
                        })

                print(_total_loss)


if __name__ == "__main__":
    create_model()
