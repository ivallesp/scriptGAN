import tensorflow as tf
import numpy as np

def decoder(inputs, thought_states, cell, max_ouput_sequence_length, name="decoder"):
    """
    Function that wraps the raw_rnn call which works similarly than dynamic_rnn.
    :param inputs: input sequence with dimensions (batch size, time steps, variables). The
    batch size and time steps can have dynamic size (3D tensor)
    :param cell: tensorflow cell. The number of units must be equal to the number of variables
    in the inputs tensor (tf cell)
    :param name: name of the scope to be used to create all the variables and tensors (string|unicode)
    """
    with tf.variable_scope(name, reuse=False):
        n_neurons = thought_states[0].get_shape()[1]
        batches_input, seq_len_input, cardinality_input = inputs.get_shape()

        embedding = tf.get_variable("embedding_in_rnn", shape=[cardinality_input, n_neurons])

        out_map_weight = tf.Variable(
            tf.random_uniform([int(n_neurons), int(cardinality_input)], minval=-0.1, maxval=0.1))
        out_map_bias = tf.Variable(tf.random_uniform([int(cardinality_input)], minval=-0.1, maxval=0.1))

        def loop_fn(time, cell_output, cell_state, loop_state):
            if cell_output is None:
                emit_output = cell_output
                next_input = input_ta.read(time)
                next_cell_state = thought_states
            else:
                cell_output = tf.nn.softmax(
                    tf.matmul(cell_output, out_map_weight) + out_map_bias)  # Same size of the input
                next_input = cell_output
                next_cell_state = cell_state
                emit_output = cell_output
            next_input = tf.matmul(next_input, embedding)
            next_loop_state = None
            finished = (time >= max_ouput_sequence_length)

            return (finished, next_input, next_cell_state, emit_output, next_loop_state)

        input_ta = tf.TensorArray(dtype=tf.float32, size=(tf.shape(inputs)[1]), clear_after_read=False)
        input_ta = input_ta.unstack(tf.transpose(inputs, [1, 0, 2]))

        emit_ta, final_state, final_loop_state = tf.nn.raw_rnn(cell, loop_fn)
        emit_ta = tf.transpose(emit_ta.stack(), [1, 0, 2])
        return (emit_ta, final_state)


def build_lstm_feed_back_layer(zh, zc, max_length, cardinality_input, name="LSTM_feed_back"):
    assert zh.shape[1] == zc.shape[1]
    with tf.variable_scope(name, reuse=False):
        cell_dec = tf.nn.rnn_cell.LSTMCell(num_units=zc.shape[1].value, )
        cell_dec._output_size = cardinality_input
        thought_states = tf.nn.rnn_cell.LSTMStateTuple(zc, zh)
        #go = tf.concat([tf.ones([tf.shape(zh)[0], max_length, cardinality_input])
        go = tf.one_hot(np.array([[1]]*int(zh.shape[0])), depth=cardinality_input)
        output_dec, states_dec = decoder(go, thought_states=thought_states, cell=cell_dec,
                                         max_ouput_sequence_length=max_length)
        return (output_dec, states_dec)