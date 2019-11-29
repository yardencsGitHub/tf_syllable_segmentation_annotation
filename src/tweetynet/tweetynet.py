"""TweetyNet model for keras"""
import tensorflow as tf


def TweetyNet(num_classes,
              input_shape=(88, 513, 1),
              conv1_filters=32,
              conv1_kernel_size=(5, 5),
              conv2_filters=64,
              conv2_kernel_size=(5, 5),
              conv_activation='relu',
              l=0.001,
              pool1_size=(1, 8),
              pool1_strides=(1, 8),
              pool2_size=(1, 8),
              pool2_strides=(1, 8),
              lstm_dropout=0.25,
              recurrent_dropout=0.1,
              ):
    """build TweetyNet model

    Parameters
    ----------
    num_classes : int
        number of classes to predict, e.g., number of syllable classes in an individual bird's song
    input_shape : tuple
        with 3 elements corresponding to dimensions of spectrogram windows: (time bins, frequency bins, channels).
        Default is (88, 513, 1).
    conv1_filters : int
        Number of filters in first convolutional layer. Default is 32.
    conv1_kernel_size : tuple
        Size of kernels, i.e. filters, in first convolutional layer. Default is (5, 5).
    conv2_filters : int
        Number of filters in second convolutional layer. Default is 64.
    conv2_kernel_size : tuple
        Size of kernels, i.e. filters, in second convolutional layer. Default is (5, 5).
    conv_activation : str
        type of activation to apply to feature maps output by convolutional layers.
        Default is 'relu'.
    l : float
        L2 regularization parameter, applied to kernels in convolutional layers.
        Default is 0.001.
    pool1_size : two element tuple of ints
        Size of sliding window for first max pooling layer. Default is (1, 8)
    pool1_strides : two element tuple of ints
        Step size for sliding window of first max pooling layer. Default is (1, 8)
    pool2_size : two element tuple of ints
        Size of sliding window for second max pooling layer. Default is (1, 8),
    pool2_strides : two element tuple of ints
        Step size for sliding window of second max pooling layer. Default is (1, 8)

    Returns
    -------
    model : tensorflow.keras.Model
        instance of TweetyNet model
    """
    inputs = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.Permute((2, 1, 3))(inputs)  # switch frequency bins and time bins axes
    x = tf.keras.layers.Conv2D(conv1_filters, kernel_size=conv1_kernel_size,
                               kernel_regularizer=tf.keras.regularizers.l2(l),
                               padding="same", activation=conv_activation)(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=pool1_size,
                                     strides=pool1_strides)(x)

    x = tf.keras.layers.Conv2D(conv2_filters, kernel_size=conv2_kernel_size,
                               kernel_regularizer=tf.keras.regularizers.l2(l),
                               padding="same", activation=conv_activation)(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=pool2_size,
                                     strides=pool2_strides)(x)

    conv_out_shape = x.get_shape().as_list()
    num_hidden = conv_out_shape[2] * conv_out_shape[3]
    new_shape = (-1, num_hidden)
    x = tf.keras.layers.Reshape(new_shape)(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(num_hidden, return_sequences=True, dropout=lstm_dropout,
                             recurrent_dropout=recurrent_dropout))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_classes, activation="softmax"))(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=x)
    return model
