import tensorflow as tf
from src.nn_frankenstein.activations import gumbel_softmax


# TODO: Fix the go symbol; putting all-ones unstabilizes the network

def decoder(inputs, thought_states, cell, max_ouput_sequence_length, gumbel_tao, name="decoder"):
    """
    Function that wraps the raw_rnn call which works similarly than dynamic_rnn.
    :param inputs: input sequence with dimensions (batch size, time steps, variables). The
    batch size and time steps can have dynamic size (3D tensor)
    :param recursion_steps: number of steps the network has to run (int)
    :param cell: tensorflow cell. The number of units must be equal to the number of variables
    in the inputs tensor (tf cell)
    :param name: name of the scope to be used to create all the variables and tensors (string|unicode)
    """
    with tf.variable_scope(name, reuse=False):
        def loop_fn(time, cell_output, cell_state, loop_state):
            if cell_output is None: # Only first time
                emit_output = cell_output
                next_input = input_ta.read(time)
                next_cell_state = thought_states
            else:
                emit_output = cell_output  # No Softmax
                next_input = emit_output
                next_cell_state = cell_state
            next_loop_state = None
            finished = (time >= max_ouput_sequence_length)
            return (finished, next_input, next_cell_state, emit_output, next_loop_state)

        input_ta = tf.TensorArray(dtype=tf.float32, size=(tf.shape(inputs)[1]), clear_after_read=False)
        input_ta = input_ta.unstack(tf.transpose(inputs, [1, 0, 2]))

        emit_ta, final_state, final_loop_state = tf.nn.raw_rnn(cell, loop_fn)
        emit_ta = tf.transpose(emit_ta.stack(), [1, 0, 2])
        return (emit_ta, final_state)


def build_lstm_feed_back_layer(zh, zc, max_length, gumbel_tao, name="LSTM_feed_back"):
    assert zh.shape[1] == zc.shape[1]
    with tf.variable_scope(name, reuse=False):
        cell_dec = tf.nn.rnn_cell.LSTMCell(num_units=zc.shape[1].value, use_peepholes=True)
        thought_states = tf.nn.rnn_cell.LSTMStateTuple(zc, zh)
        go = tf.ones([tf.shape(zh)[0], max_length, zh.shape[1].value])
        output_dec, states_dec = decoder(go, thought_states=thought_states, cell=cell_dec,
                                         max_ouput_sequence_length=max_length, gumbel_tao=gumbel_tao)
        return (output_dec, states_dec)
