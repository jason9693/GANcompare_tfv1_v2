import tensorflow as tf
from util import tf_utils, processing
import pprint
import numpy as np
import params as par

class GAN:  #Vanilla GAN Model compatible with tf1.x
    def __init__(self, input_shape, learning_rate,noise_dim, num_classes=1, sess=None, ckpt_path=None, net='gan'):
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.sess= sess
        self.noise_dim = noise_dim
        self.net = net
        self.__build_net__()

        if ckpt_path is not None:
            saver = tf.train.Saver()
            saver.restore(sess=self.sess, save_path=ckpt_path)
        else:
            self.sess.run(tf.global_variables_initializer())


    def __build_net__(self):
        with tf_utils.set_device_mode(par.gpu_mode):
            self.X = tf.placeholder(shape=[None]+self.input_shape, dtype=tf.float32, name='X')
            self.Z = tf.placeholder(shape=[None, self.noise_dim], dtype=tf.float32, name='random_z')

            self.G = self.Generator(self.Z , self.X.shape[1])

            self.D = self.Discriminator(self.X, self.num_classes)
            self.D_G = self.Discriminator(self.G, self.num_classes)

            self.__set_loss_and_optim__()
        return

    def Discriminator(self, input, output_dim, name = 'discriminator'):
        with tf.variable_scope('discriminator',reuse=tf.AUTO_REUSE):
            L1 = tf_utils.Dense(input, 256, name=name+'/L1', activation=tf.nn.leaky_relu)
            L2 = tf_utils.Dense(L1, 256, name=name+'/L2', activation=tf.nn.leaky_relu)
            L3 = tf_utils.Dense(L2, output_dim, name=name+'/L3', activation=None)
        return L3

    def Generator(self,z , output_dim, name= 'generator'):
        L1 = tf_utils.Dense(z, z.shape[1] // 2, name=name+'/L1', activation=tf.nn.relu)
        L2 = tf_utils.Dense(L1, z.shape[1], name=name + '/L2', activation=tf.nn.relu)
        L3 = tf_utils.Dense(L2, output_dim, name=name + '/L3', activation=None)
        return tf.tanh(L3)

    def __set_loss_and_optim__(self):
        logits_real = tf.ones_like(self.D_G)
        logits_fake = tf.zeros_like(self.D_G)

        self.G_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_G, labels=logits_real)
        self.G_loss = tf.reduce_mean(self.G_loss)

        self.D_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D, labels=logits_real) \
        + tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_G, labels=logits_fake)
        self.D_loss = tf.reduce_mean(self.D_loss)

        D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
        G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')

        print('Discriminator variables: ',D_vars)
        print('\nGenerator variables: ', G_vars)

        self.D_optim = \
            tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.D_loss, var_list=D_vars)
        self.G_optim = \
            tf.train.AdamOptimizer(self.learning_rate).minimize(self.G_loss, var_list=G_vars)
        return

    def train(self, x= None, z=None, y=None):
        if x is None:
            return self.sess.run([self.G_optim, self.G_loss], feed_dict = {
               self.Z: z
            })[1]
        else:
            x = processing.img_preprocessing(x)
            return self.sess.run([self.D_optim, self.D_loss], feed_dict = {
                self.X: x,
                self.Z: z
            })[1]

    def eval(self, z, y=None):
        out = self.sess.run(self.G, feed_dict = {self.Z: z})
        out = processing.img_deprocessing(out)
        fig = processing.show_images(out,'generated/save.png')
        return fig

    def infer(self, z, y=None, path=None):
        fig = self.eval(z)
        fig.savefig('generated/{}.png'.format(self.net))
        return


class GANv2:    #compatible with tf2.x
    def __init__(self, input_shape, learning_rate,noise_dim, num_classes=1, sess=None, ckpt_path=None, net='gan'):
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.sess= sess
        self.noise_dim = noise_dim
        self.net = net
        self.__build_net__()

        if ckpt_path is not None:
            saver = tf.train.Saver()
            saver.restore(sess=self.sess, save_path=ckpt_path)
        else:
            self.sess.run(tf.global_variables_initializer())

    def __build_net__(self):
        with tf_utils.set_device_mode(par.gpu_mode):
            self.X = tf.placeholder(shape=[None]+self.input_shape, dtype=tf.float32, name='X')
            self.Z = tf.placeholder(shape=[None, self.noise_dim], dtype=tf.float32, name='random_z')

            self.modelG = self.Generator(self.Z.shape[1:] , self.X.shape[1])
            self.modelD = self.Discriminator(self.X.shape[1:], self.num_classes)

            self.G = self.modelG(self.Z)

            self.D = self.modelD(self.X)
            self.D_G = self.modelD(self.G)

            self.__set_loss_and_optim__()
        return

    def Discriminator(self, input_shape, output_dim, name = 'discriminator'):
        inputs = tf.keras.Input(input_shape)
        L1 = tf.keras.layers.Dense(256, name='L1', activation=tf.nn.leaky_relu)(inputs)
        L2 = tf.keras.layers.Dense(256, name='L2', activation=tf.nn.leaky_relu)(L1)
        L3 = tf.keras.layers.Dense(output_dim, name=name, activation=None)(L2)
        if int(tf.__version__.split('.')[0]) < 2:
            D = tf.keras.Model(inputs, L3)
        else:
            D = tf.keras.Model(inputs=inputs, output = L3)
        return D

    def Generator(self,z_shape, output_dim, name= 'generator'):
        inputs = tf.keras.Input(z_shape)
        L1 = tf.keras.layers.Dense(z_shape[0] // 2, name='L1', activation=tf.nn.relu)(inputs)
        L2 = tf.keras.layers.Dense(z_shape[0], name='L2', activation=tf.nn.relu)(L1)
        L3 = tf.keras.layers.Dense(output_dim, name=name, activation=tf.nn.tanh)(L2)
        if int(tf.__version__.split('.')[0]) < 2:
            G = tf.keras.Model(inputs, L3)
        else:
            G = tf.keras.Model(inputs=inputs, output=L3)
        return G

    def __set_loss_and_optim__(self):
        logits_real = tf.ones_like(self.D_G)
        logits_fake = tf.zeros_like(self.D_G)

        self.G_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_G, labels=logits_real)
        self.G_loss = tf.reduce_mean(self.G_loss)

        self.D_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D, labels=logits_real) \
        + tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_G, labels=logits_fake)
        self.D_loss = tf.reduce_mean(self.D_loss)

        D_vars = self.modelD.trainable_variables
        G_vars = self.modelG.trainable_variables

        self.D_optim = \
            tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.D_loss, var_list=D_vars)
        self.G_optim = \
            tf.train.AdamOptimizer(self.learning_rate).minimize(self.G_loss, var_list=G_vars)
        return

    def train(self, x= None, z=None, y=None):
        if x is None:
            return self.sess.run([self.G_optim, self.G_loss], feed_dict = {
               self.Z: z
            })[1]
        else:
            x = processing.img_preprocessing(x)
            return self.sess.run([self.D_optim, self.D_loss], feed_dict = {
                self.X: x,
                self.Z: z
            })[1]

    def eval(self, z, y=None):
        out = self.sess.run(self.G, feed_dict = {self.Z: z})
        out = processing.img_deprocessing(out)
        fig = processing.show_images(out,'generated/save.png')

        return fig

    def infer(self, z, y=None, path=None):
        fig = self.eval(z)
        fig.savefig('generated/{}.png'.format(self.net))
        return
