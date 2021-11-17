import tensorflow as tf


class Discriminator:
    def __init__(self, env):
        """
        :param env:
        Output of this Discriminator is reward for learning agent. Not the cost.
        Because discriminator predicts  P(expert|s,a) = 1 - P(agent|s,a).
        """

        ob_space = env.reset()
        action_space_size = 3
        with tf.variable_scope('discriminator'):
            self.scope = tf.get_variable_scope().name
            self.expert_s = tf.placeholder(dtype=tf.float32, shape=[None, ob_space.shape[1], ob_space.shape[2], ob_space.shape[3]])
            # self.expert_a = tf.placeholder(dtype=tf.int32, shape=(action_space_size, 1))

            # self.expert_a = tf.placeholder(dtype=tf.int32, shape=(None, action_space_size))
            # # expert_a_one_hot = tf.one_hot(self.expert_a, depth=action_space_size)
            # # add noise for stabilise training
            # self.expert_a  += tf.random_normal(tf.shape(self.expert_a), mean=0.2, stddev=0.1, dtype=tf.float32)/1.2
            # expert_s_a = tf.concat([self.expert_s, self.expert_a], axis=1)


            # self.agent_s = tf.placeholder(dtype=tf.float32, shape=list(ob_space.shape) + [1]) #shape=(None, env.observation_space.shape[0], env.observation_space.shape[1]))
            self.agent_s = tf.placeholder(dtype=tf.float32, shape=[None, ob_space.shape[1], ob_space.shape[2], ob_space.shape[3]])

            # self.agent_a = tf.placeholder(dtype=tf.int32, shape=(None, action_space_size))

            # # agent_a_one_hot = tf.one_hot(self.agent_a, depth=action_space_size)
            # # # add noise for stabilise training
            # self.agent_a += tf.random_normal(tf.shape(self.agent_a), mean=0.2, stddev=0.1, dtype=tf.float32)/1.2
            # agent_s_a = tf.concat([self.agent_s, self.agent_a], axis=1)

            with tf.variable_scope('network') as network_scope:
                prob_1 = self.construct_network(input=self.expert_s)
                network_scope.reuse_variables()  # share parameter
                prob_2 = self.construct_network(input=self.agent_s)

            with tf.variable_scope('loss'):
                loss_expert = tf.reduce_mean(tf.log(tf.clip_by_value(prob_1, 0.01, 1)))
                loss_agent = tf.reduce_mean(tf.log(tf.clip_by_value(1 - prob_2, 0.01, 1)))
                loss = loss_expert + loss_agent
                loss = -loss
                tf.summary.scalar('discriminator', loss)

            optimizer = tf.train.AdamOptimizer()
            self.train_op = optimizer.minimize(loss)

            self.rewards = tf.log(tf.clip_by_value(prob_2, 1e-10, 1))  # log(P(expert|s,a)) larger is better for agent

    def construct_network(self, input):
        # layer_1 = tf.layers.dense(inputs=input, units=20, activation=tf.nn.leaky_relu, name='layer1')
        # layer_2 = tf.layers.dense(inputs=layer_1, units=20, activation=tf.nn.leaky_relu, name='layer2')
        # layer_3 = tf.layers.dense(inputs=layer_2, units=20, activation=tf.nn.leaky_relu, name='layer3')
        # prob = tf.layers.dense(inputs=layer_3, units=1, activation=tf.sigmoid, name='prob')

        # layer_1 = tf.layers.dense(inputs=self.obs, units=20, activation=tf.tanh)
        layer_1 = tf.layers.conv2d(inputs=input, filters=8, kernel_size=4, strides=2, activation=tf.nn.relu, name='layer1')
        # layer_2 = tf.layers.dense(inputs=layer_1, units=20, activation=tf.tanh)
        layer_2 = tf.layers.conv2d(inputs=layer_1, filters=16, kernel_size=3, strides=2, activation=tf.nn.relu, name='layer2')
        layer_3 = tf.layers.conv2d(inputs=layer_2, filters=32, kernel_size=3, strides=2, activation=tf.nn.relu, name='layer3')
        layer_4 = tf.layers.conv2d(inputs=layer_3, filters=64, kernel_size=3, strides=2, activation=tf.nn.relu, name='layer4')
        layer_5 = tf.layers.conv2d(inputs=layer_4, filters=128, kernel_size=3, strides=1, activation=tf.nn.relu,name='layer5')
        layer_6 = tf.layers.conv2d(inputs=layer_5, filters=256, kernel_size=3, strides=1, activation=tf.nn.relu,name='layer6')

        layer_7 = tf.layers.Flatten()(layer_6)  # not using this right now
        layer_8 = tf.layers.dense(inputs=layer_7, units=100, activation=tf.nn.relu, name='layer8')
        layer_9 = tf.layers.dense(inputs=layer_8, units=1, activation=tf.sigmoid, name='layer9')

        return layer_9


    def train(self, expert_s, agent_s, expert_a, agent_a):
        return tf.get_default_session().run(self.train_op, feed_dict={self.expert_s: expert_s,
                                                                      self.agent_s: agent_s})

    def get_rewards(self, agent_s, agent_a):
        # return tf.get_default_session().run(self.rewards, feed_dict={self.agent_s: agent_s,
        #                                                              self.agent_a: agent_a})

        return tf.get_default_session().run(self.rewards, feed_dict={self.agent_s: agent_s})

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

