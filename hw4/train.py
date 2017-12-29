#-*- coding:utf-8 -*-

# -------------------------------------------------------------------- #
#                                AC-GAN                                #
# Deep Convolution Auxiliary Classifier Generative Adversarial Network #
# -------------------------------------------------------------------- #
#                             References                               #
#             https://github.com/eriklindernoren/Keras-GAN             #
#             https://github.com/lukedeo/keras-acgan                   #
#             https://github.com/soumith/ganhacks                      #
# -------------------------------------------------------------------- #

import os, sys, csv, time, numpy, pandas, skimage.io, skimage.transform
from collections import OrderedDict
from scipy.misc  import imsave

import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.optimizers import Adam

# ------------------------------- tags ------------------------------- #
hair_color = [   'red hair', 'blonde hair', 'orange hair',
               'green hair',   'aqua hair',   'blue hair',
                'pink hair', 'purple hair',  'brown hair',
               'white hair',   'gray hair',  'black hair']

eye_color  = [   'red eyes',                'orange eyes', 'yellow eyes',
               'green eyes',   'aqua eyes',   'blue eyes',
                'pink eyes', 'purple eyes',  'brown eyes',
                               'gray eyes',  'black eyes']
# --------------------------------------------------------------------- #
def prepro_tag( tag_path ):
    '''
    training tags file format:
    'img_id' <,> 'tag1' <:> '#_post' <tab> 'tag2' <:> '#_post' <tab> ......

    testing  text file format:
    'testing_text_id' <,> 'testing_text'
    '''
    tags = pandas.read_csv( tag_path, sep=',', dtype=str, header=None).values
    hair = numpy.zeros( ( tags.shape[0], len(hair_color) ), dtype=numpy.float)
    eye  = numpy.zeros( ( tags.shape[0], len( eye_color) ), dtype=numpy.float)
    for i in range(tags.shape[0]):
        for h in range( len( hair_color ) ):
            if tags[i][1].find( hair_color[h] ) > (-1):
                hair[i][h] = 1
        for e in range( len( eye_color ) ):
            if tags[i][1].find( eye_color[e] ) > (-1):
                eye[i][e] = 1
    return tags[:,0], hair, eye

class ACGAN():
    def __init__(self):
        self.img_rows  =  96
        self.img_cols  =  96
        self.channels  =   3
        self.noise_dim = 100
        # self.noise_sigma = 0.01
        self.sigma     =   1
        self.d_loss_weights = [ 1, 0.3, 0.35]
        self.g_loss_weights = [ 1, 0.3, 0.35]
        self.hair_color_dim = len(hair_color)
        self.eye_color_dim  = len( eye_color)
        self.image_dir = '../faces'
        self.tag_path  = './tags_clean.csv'
        self.model_dir = './model/'
        self.output_dir = './samples/'

        optimizer = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(optimizer=optimizer,
                                   loss=['binary_crossentropy',
                                         'categorical_crossentropy',
                                         'categorical_crossentropy'],
                                   loss_weights=self.d_loss_weights,
                                   metrics=['accuracy'])
        self.discriminator.summary()

        self.generator = self.build_generator()
        self.generator.compile(optimizer=optimizer, loss='binary_crossentropy')
        self.generator.summary()

        # generator - input: [noise, tags], output: imgs
        z        = Input( shape=( self.noise_dim, ) )
        hair_tag = Input( shape=( self.hair_color_dim, ) )
        eye_tag  = Input( shape=( self.eye_color_dim,  ) )
        img = self.generator([z, hair_tag, eye_tag])

        # only train the generator for combined model
        '''
        you may ignore the warning below, since the trainable discriminator is already compiled
        UserWarning: Discrepancy between trainable weights and collected trainable weights, did you set `model.trainable` without calling `model.compile` after ?
        'Discrepancy between trainable weights and collected trainable'
        '''
        self.discriminator.trainable = False

        # valid takes generated images as input and determines validity
        # *_class determines validity of tags
        [valid, hair_class, eye_class] = self.discriminator(img)

        # The combined model  (stacked generator and discriminator) takes
        # [noise, tags] as input => generates images => determines validity 
        self.combined = Model(inputs=[z, hair_tag, eye_tag], outputs=[valid, hair_class, eye_class])
        self.combined.compile(optimizer=optimizer,
                              loss=['binary_crossentropy',
                                    'categorical_crossentropy',
                                    'categorical_crossentropy'],
                              loss_weights=self.g_loss_weights )

    def build_generator(self):
        noise    = Input( shape=( self.noise_dim,      ) )
        hair_tag = Input( shape=( self.hair_color_dim, ) )
        eye_tag  = Input( shape=( self.eye_color_dim,  ) )
        merged   = Concatenate()([noise, hair_tag, eye_tag])

        g_model = Dense(128 * 6 * 6, activation='relu')( merged )
        g_model = Reshape((6, 6, 128))( g_model )
        
        g_model = BatchNormalization(momentum=0.75)( g_model )
        g_model = UpSampling2D()( g_model )
        g_model = Conv2D( 512, kernel_size=3, padding='same', activation='relu')( g_model )

        g_model = BatchNormalization(momentum=0.75)( g_model ) 
        g_model = UpSampling2D()( g_model )
        g_model = Conv2D( 256, kernel_size=3, padding='same', activation='relu')( g_model )

        g_model = BatchNormalization(momentum=0.75)( g_model )
        g_model = UpSampling2D()( g_model )
        g_model = Conv2D( 128, kernel_size=3, padding='same', activation='relu')( g_model )

        g_model = BatchNormalization(momentum=0.75)( g_model )
        g_model = UpSampling2D()( g_model )
        g_model = Conv2D(  64, kernel_size=3, padding='same', activation='relu')( g_model )

        g_model = BatchNormalization(momentum=0.75)( g_model )
        img     = Conv2D(self.channels, kernel_size=3, padding='same', activation='tanh')( g_model )

        return Model(inputs=[noise, hair_tag, eye_tag], outputs=img)

    def build_discriminator(self):
        img = Input( shape=( self.img_rows, self.img_cols, self.channels) )

        d_model = Conv2D(  32, kernel_size=3, strides=2, padding='same')( img )
        d_model = LeakyReLU(alpha=0.2)( d_model )
        d_model = Dropout(0.25)( d_model )

        d_model = Conv2D(  64, kernel_size=3, strides=1, padding='same')( d_model )
        d_model = ZeroPadding2D(padding=((0,1),(0,1)))( d_model )
        d_model = LeakyReLU(alpha=0.2)( d_model )
        d_model = Dropout(0.25)( d_model )

        d_model = BatchNormalization(momentum=0.75)( d_model )
        d_model = Conv2D( 128, kernel_size=3, strides=2, padding='same')( d_model )
        d_model = LeakyReLU(alpha=0.2)( d_model )
        d_model = Dropout(0.25)( d_model )

        d_model = BatchNormalization(momentum=0.75)( d_model )
        d_model = Conv2D( 256, kernel_size=3, strides=1, padding='same')( d_model )
        d_model = LeakyReLU(alpha=0.2)( d_model )
        d_model = Dropout(0.5)( d_model )

        d_model = BatchNormalization(momentum=0.75)( d_model )
        d_model = Conv2D( 256, kernel_size=3, strides=2, padding='same')( d_model )
        d_model = LeakyReLU(alpha=0.2)( d_model )
        d_model = Dropout(0.5)( d_model )

        d_model = Flatten()( d_model )

        validity   = Dense(                  1, activation='sigmoid')( d_model )
        hair_class = Dense(self.hair_color_dim, activation='softmax')( d_model )
        eye_class  = Dense( self.eye_color_dim, activation='softmax')( d_model )

        return Model(inputs=img, outputs=[validity, hair_class, eye_class])

    def train(self, batch_size=64, save_interval=1000):
        # limiting GPU memory usage
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        # read in real images & tags
        X_train    = []
        X_hair_tag = []
        X_eye_tag  = []
        IDs, hair_tag, eye_tag = prepro_tag( self.tag_path )
        for i in range(IDs.shape[0]):
            # if numpy.count_nonzero(hair_tag[i]) == 1 and numpy.count_nonzero(eye_tag[i]) == 1:
            if True:
                X_hair_tag.append(hair_tag[i])
                X_eye_tag.append(eye_tag[i])
                X_train.append( skimage.io.imread( os.path.join( self.image_dir, str(IDs[i]) + '.jpg')))

        X_train    = numpy.asarray(X_train   , dtype=numpy.float)
        X_hair_tag = numpy.asarray(X_hair_tag, dtype=numpy.float)
        X_eye_tag  = numpy.asarray(X_eye_tag , dtype=numpy.float)
        # rescale to -1 ~ 1
        X_train = (X_train - 127.5) / 127.5
        print('__________________________________________________________________\n')
        print('X_train(samples, img_rows, img_cols, channels):', X_train.shape)
        print('X_hair_tag(samples, len(hair_color)):', X_hair_tag.shape)
        print('X_eye_tag(samples, len(eye_color)):', X_eye_tag.shape)
        print('__________________________________________________________________')
        epoch = 0
        while True:
            epoch += 1
            # Select a random batch of images
            idx = numpy.random.randint(0, X_train.shape[0], batch_size)
            true_imgs      = numpy.copy( X_train[idx] )
            # add noise to input (optional)
            # true_imgs     += numpy.random.normal( 0, self.noise_sigma, (true_imgs.shape[0], self.img_rows, self.img_cols, self.channels))
            true_hair_tags = numpy.copy( X_hair_tag[idx] )
            true_eye_tags  = numpy.copy( X_eye_tag[idx]  )
            
            for choose in range(3):
                # Sample noise
                noise = numpy.random.normal( 0, self.sigma, (true_imgs.shape[0], self.noise_dim))
                rdm_hair_tags = numpy.zeros( (true_imgs.shape[0], self.hair_color_dim), dtype=numpy.float)
                rdm_eye_tags  = numpy.zeros( (true_imgs.shape[0], self.eye_color_dim ), dtype=numpy.float)
                for i in range(true_imgs.shape[0]):
                    rdm_hair_tags[i][numpy.random.randint( len(hair_color) )] = 1
                    rdm_eye_tags[i][ numpy.random.randint( len(eye_color)  )] = 1

                if choose == 0:
                    # train Discriminator
                    # generate a batch of fake images
                    gen_imgs = self.generator.predict([noise, rdm_hair_tags, rdm_eye_tags])
                    # train the discriminator (real classified as ones and generated as zeros)
                    d_loss_real  = self.discriminator.train_on_batch(true_imgs, [numpy.ones(  (true_imgs.shape[0], 1)), true_hair_tags, true_eye_tags])
                    d_loss_fake  = self.discriminator.train_on_batch( gen_imgs, [numpy.zeros( (true_imgs.shape[0], 1)),
                                                                                 numpy.zeros( (true_imgs.shape[0], self.hair_color_dim)),
                                                                                 numpy.zeros( (true_imgs.shape[0],  self.eye_color_dim))])
                    d_loss = 0.5 * numpy.add(d_loss_real, d_loss_fake)

                else:
                    # train Generator
                    # train the generator (wants discriminator to mistake images as real)
                    g_loss = self.combined.train_on_batch([noise, rdm_hair_tags, rdm_eye_tags],
                                                          [numpy.ones((true_imgs.shape[0], 1)), rdm_hair_tags, rdm_eye_tags])

            # print loss & acc
            print ('%d, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f'
                   % (epoch, d_loss[1], d_loss[2], d_loss[3],
                             d_loss[4], d_loss[5], d_loss[6],
                             g_loss[1], g_loss[2], g_loss[3]))
            sys.stdout.flush()
            # output image samples & save model
            if epoch % save_interval == 0:
                self.save_imgs(epoch)
                self.discriminator.save( os.path.join( self.model_dir, str(epoch) + '_dis.h5' ) )
                self.generator.save(     os.path.join( self.model_dir, str(epoch) + '_gen.h5' ) )

    def save_imgs(self, epoch):
        noise = numpy.random.normal(   0, self.sigma, (32, self.noise_dim))
        rdm_hair_tags = numpy.zeros( (32, self.hair_color_dim), dtype=numpy.float)
        rdm_eye_tags  = numpy.zeros( (32, self.eye_color_dim) , dtype=numpy.float)

        gen_imgs = self.generator.predict([noise, rdm_hair_tags, rdm_eye_tags])

        # rescale images to 0 ~ 255
        gen_imgs = 127.5 * gen_imgs + 127.5
        gen_imgs = gen_imgs.astype(dtype=numpy.uint8)
        for i in range(gen_imgs.shape[0]):
            imsave( self.output_dir + str(epoch) + ' ' + str(hair_color[numpy.nonzero(rdm_hair_tags[i])[0][0]])
                   + ' ' + str(eye_color[numpy.nonzero(rdm_eye_tags[i])[0][0]]) + '.jpg', gen_imgs[i])

if __name__ == '__main__':
    # you may add argv parsing if you needed
    gan = ACGAN()
    gan.train()
