import numpy as np
import tensorflow as tf

class AutomataModel(tf.keras.Model):
    """ Learned update rule model

    Attributes:
        state_size: State size for each cell
        drop_prob: Chance to randomly mask cells. Defaults to 0.5
        filters (tensor): Stack of perception filters, of shape [HEIGHT, WIDTH,
            STATE_SIZE, 3]
        dense (keras.Conv2D): First dense layer for each pixel, implemented as
            a convlutional layer with 128 filers
        out (keras.Conv2D): Output dense layer for each pixel, implemented as
            as convolutional layer with 16 filters
    """
    def __init__(self, state_size, drop_prob=0.5):
        super().__init__()
        self.state_size = state_size
        self.drop_prob = drop_prob
        assert self.state_size >= 4, "Must have state size of at least 4!"

        self.filters = self.build_filters()
        self.filters.trainable = False
        self.dense = tf.keras.layers.Conv2D(
            filters=48,
            kernel_size=1,
            padding="Same",
            activation=tf.keras.activations.relu
        )
        self.out = tf.keras.layers.Conv2D(
            filters=self.state_size,
            kernel_size=1,
            padding="Same",
            kernel_initializer=tf.zeros_initializer
        )

    # @tf.function
    def build_filters(self):
        """ Builds 2 sobel and 1 identiy filter  and formats """
        identity = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=np.float32) / 8
        sobel_y = np.transpose(sobel_x)
        filters = tf.stack([identity, sobel_x, sobel_y], axis=-1)
        # shape [kernel height, kernel width, num in, num out per in]
        filters = tf.repeat(filters[:, :, None, :], self.state_size, axis=2)
        return filters

    # @tf.function
    def build_stochastic_mask(self, cell_state):
        """ Builds stochastic mask, randomly drops cells with DROP_PROB

        Args:
            cell_state (tensor): Array or tensor of shape [batch size, height,
                width, state size]
        """
        mask = tf.random.uniform(tf.shape(cell_state[:, :, :, :1]))
        mask = tf.cast(mask <= self.drop_prob, tf.float32)
        return mask

    # @tf.function
    def build_live_mask(self, cell_state):
        """ Builds live mask, drops cells without living neighbhour

        Args:
            cell_state (tensor): Array or tensor of shape [batch size, height,
                width, state size] (4th value in state is assumed to be alpha)
        """
        mask = tf.nn.max_pool(
            cell_state[:, :, :, 3:4],
            ksize=3,
            strides=1,
            padding="SAME"
        )
        mask = mask >= 0.1
        return mask

    # @tf.function
    def call(self, cell_state):
        """ Updates the cell state a single step

        Args:
            cell_state (tensor): Array or tensor of shape [batch size, height,
                width, state size]
        """
        pre_life_mask = self.build_live_mask(cell_state)

        update_vec = tf.nn.depthwise_conv2d(
            cell_state,
            self.filters,
            [1, 1, 1, 1],
            padding="SAME"
        )
        update_vec = self.dense(update_vec)
        update_vec = self.out(update_vec)

        stochastic_mask = self.build_stochastic_mask(cell_state)
        cell_state += update_vec * stochastic_mask

        post_life_mask = self.build_live_mask(cell_state)
        # Mask if cell doesn't neighbhour alive cell before and after update
        life_mask = pre_life_mask & post_life_mask
        cell_state *= tf.cast(life_mask, tf.float32)
        return cell_state
