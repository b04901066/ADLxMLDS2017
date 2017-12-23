import os, sys, csv, time, numpy, pandas
from collections import OrderedDict
from scipy.misc import imsave
import skimage.io
import skimage.transform

import tensorflow as tf
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.optimizers import Adam

hair_color = [   'red hair', 'blonde hair', 'orange hair',
               'green hair',   'aqua hair',   'blue hair',
                'pink hair', 'purple hair',  'brown hair',
               'white hair',   'gray hair',  'black hair']

eye_color  = [   'red eyes',                'orange eyes', 'yellow eyes',
               'green eyes',   'aqua eyes',   'blue eyes',
                'pink eyes', 'purple eyes',  'brown eyes',
                               'gray eyes',  'black eyes']

overall_color = [   'red', 'blonde', 'orange', 'yellow',
                  'green',   'aqua',   'blue',
                   'pink', 'purple',  'brown',
                  'white',   'gray',  'black']

def prepro_tag( tag_path ):
    tags = pandas.read_csv( tag_path, sep=',', dtype=str, header=None).values
    color = numpy.zeros( ( tags.shape[0], len(hair_color) + len(eye_color)), dtype=numpy.float)
    for i in range(tags.shape[0]):
        for h in range( len( hair_color ) ):
            if tags[i][1].find( hair_color[h] ) > (-1):
                color[i][h] = 1
        for e in range( len( eye_color ) ):
            if tags[i][1].find( eye_color[e] ) > (-1):
                color[i][len(hair_color) + e] = 1
        '''
        if numpy.count_nonzero( color[ i, 0:len(hair_color)] ) == 0:
            for c in range( len( overall_color ) ):
                if tags[i][1].find( overall_color[c] ) > (-1):
                    for h in range( len( hair_color ) ):
                        if ( overall_color[c]+' hair').find( hair_color[h] ) > (-1):
                            color[i][h] = 1
                    break
        if numpy.count_nonzero( color[ i, len(hair_color):len(hair_color) + len(eye_color)] ) == 0:
            for c in range( len( overall_color ) ):
                if tags[i][1].find( overall_color[c] ) > (-1):
                    for e in range( len( eye_color ) ):
                        if ( overall_color[c]+' eyes').find( eye_color[e] ) > (-1):
                            color[i][len(hair_color) + e] = 1
                    break
        '''
    return tags[:,0], color

# -------------------------------------------- #
#                  Reference                   #
# https://github.com/eriklindernoren/Keras-GAN #
# -------------------------------------------- #
class DCGAN():
    def __init__(self):
        self.img_rows  =  96
        self.img_cols  =  96
        self.channels  =   3
        self.noise_dim = 100
        self.color_dim = len(hair_color) + len(eye_color)
        self.image_dir = '../faces'
        self.tag_path  = './tags_clean.csv'
        self.discriminator_path = './gan_dis.h5'
        self.generator_path = './gan_gen.h5'

        optimizer = Adam(lr=0.0002, beta_1=0.5)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        self.discriminator.summary()

        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)
        self.generator.summary()

        # The generator takes noise as input and generated imgs
        z = Input( shape=( self.noise_dim + self.color_dim, ) )
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        tag = Input( shape=( self.color_dim , ) )
        valid = self.discriminator([img, tag])

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity 
        self.combined = Model(inputs=[z, tag], outputs=valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):
    
        input_shape = ( self.noise_dim + self.color_dim , )
        
        model = Sequential()
        model.add( Dense(128 * 6 * 6, activation='relu', input_shape=input_shape) )
        model.add( Reshape((6, 6, 128)) )
        model.add( BatchNormalization(momentum=0.75) )
        model.add( UpSampling2D() )
        model.add( Conv2D(128, kernel_size=3, padding='same', activation='relu') )
        model.add( BatchNormalization(momentum=0.75) ) 
        model.add( UpSampling2D() )
        model.add( Conv2D(128, kernel_size=3, padding='same', activation='relu') )
        model.add( BatchNormalization(momentum=0.75) )
        model.add( UpSampling2D() )
        model.add( Conv2D(64, kernel_size=3, padding='same', activation='relu') )
        model.add( BatchNormalization(momentum=0.75) )
        model.add( UpSampling2D() )
        model.add( Conv2D(64, kernel_size=3, padding='same', activation='relu') )
        model.add( BatchNormalization(momentum=0.75) )
        model.add( Conv2D(self.channels, kernel_size=3, padding='same', activation='tanh') )
        model.summary()
        
        noise_tag = Input(shape=input_shape)
        img = model(noise_tag)
        return Model(inputs=noise_tag, outputs=img)

    def build_discriminator(self):
        img = Input( shape=( self.img_rows, self.img_cols, self.channels))
        tag = Input( shape=( self.color_dim, ) )

        m1 = Conv2D(  32, kernel_size=3, strides=2, padding='same')(img)
        m1 = LeakyReLU(alpha=0.2)(m1)
        m1 = Dropout(0.25)(m1)

        m1 = Conv2D(  64, kernel_size=3, strides=2, padding='same')(m1)
        m1 = ZeroPadding2D(padding=((0,1),(0,1)))(m1)
        m1 = LeakyReLU(alpha=0.2)(m1)
        m1 = Dropout(0.25)(m1)

        m1 = BatchNormalization(momentum=0.75)(m1)
        m1 = Conv2D(  64, kernel_size=3, strides=2, padding='same')(m1)
        m1 = LeakyReLU(alpha=0.2)(m1)
        m1 = Dropout(0.25)(m1)

        m1 = BatchNormalization(momentum=0.75)(m1)
        m1 = Conv2D( 128, kernel_size=3, strides=2, padding='same')(m1)
        m1 = LeakyReLU(alpha=0.2)(m1)
        m1 = Dropout(0.25)(m1)

        m1 = BatchNormalization(momentum=0.75)(m1)
        m1 = Conv2D( 256, kernel_size=3, strides=2, padding='same')(m1)
        m1 = LeakyReLU(alpha=0.2)(m1)
        m1 = Dropout(0.25)(m1)

        m1 = Flatten()(m1)

        m1 = Dense(  64)(m1)
        m1 = LeakyReLU(alpha=0.2)(m1)
        m1 = Dropout(0.25)(m1)

        m2 = Concatenate()([m1, tag])
        validity = Dense(1, activation='sigmoid')(m2)

        return Model(inputs=[img, tag], outputs=validity)

    def train(self, batch_size=64, save_interval=50):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.35)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        # Load the dataset
        # (X_train, _), (_, _) = mnist.load_data()
        #X_train = numpy.zeros((len(imd_paths), self.img_rows, self.img_cols, self.channels), dtype=numpy.float)
        X_train = []
        X_tag = []
        IDs, color_tag = prepro_tag( self.tag_path )
        for i in range(IDs.shape[0]):
            if numpy.count_nonzero(color_tag[i]) > 0:
                X_tag.append(color_tag[i])
                X_train.append(skimage.io.imread( os.path.join( self.image_dir, str(IDs[i]) + '.jpg')))

        X_train = numpy.asarray(X_train, dtype=numpy.float)
        X_tag   = numpy.asarray(X_tag  , dtype=numpy.float)
        # Rescale 0 ~ 1
        X_train = (X_train - 127.5) / 127.5
        print('__________________________________________________________________\n')
        print('X_train(samples, img_rows, img_cols, channels):', X_train.shape)
        print('X_tag(samples, len(hair_color) + len(eye_color)):', X_tag.shape)
        print('__________________________________________________________________')
        epoch = 0
        while True:
            epoch += 1
            # --------------------- #
            #  Train Discriminator  #
            # --------------------- #

            true_imgs = X_train
            true_tags = X_tag

            # Sample noise and generate a batch of new images
            noise = numpy.random.normal(0, 1, (true_imgs.shape[0], self.noise_dim))
            rdm_tags = numpy.zeros( (true_imgs.shape[0], self.color_dim), dtype=numpy.float)
            for i in range(true_imgs.shape[0]):
                rdm_tags[i][numpy.random.randint( len(hair_color) )] = 1
                rdm_tags[i][len(hair_color) + numpy.random.randint( len(eye_color) )] = 1

            gen_imgs = self.generator.predict(numpy.append(noise, rdm_tags, axis=1))

            train_imgs = numpy.concatenate( (true_imgs, true_imgs, gen_imgs), axis=0)
            train_tags = numpy.concatenate( (true_tags,  rdm_tags, rdm_tags), axis=0)
            train_labs = numpy.concatenate( (numpy.ones( (true_imgs.shape[0], 1)),
                                             numpy.zeros((true_imgs.shape[0], 1)),
                                             numpy.zeros((true_imgs.shape[0], 1))), axis=0)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_history = self.discriminator.fit([train_imgs,train_tags], train_labs, batch_size=batch_size, epochs=1, verbose=0)

            # --------------------- #
            #    Train Generator    #
            # --------------------- #

            noise = numpy.random.normal(0, 1, (true_imgs.shape[0], self.noise_dim))
            rdm_tags = numpy.zeros( (true_imgs.shape[0], self.color_dim), dtype=numpy.float)
            for i in range(true_imgs.shape[0]):
                rdm_tags[i][numpy.random.randint( len(hair_color) )] = 1
                rdm_tags[i][len(hair_color) + numpy.random.randint( len(eye_color) )] = 1

            # Train the generator (wants discriminator to mistake images as real)
            g_history = self.combined.fit([numpy.append(noise, rdm_tags, axis=1), rdm_tags], numpy.ones((true_imgs.shape[0], 1)), batch_size=batch_size, epochs=1, verbose=0)

            # Plot the progress
            print ('Epoch %d | Dis - loss: %.4f - acc: %.4f | Gen - loss: %.4f' % (epoch, d_history.history['loss'][0], d_history.history['acc'][0], g_history.history['loss'][0]))
            sys.stdout.flush()
            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)
                self.discriminator.save( str(epoch) + self.discriminator_path )
                self.generator.save( str(epoch) + self.generator_path )
    
    def test(self, testing_text_path):
        # testing_text_id <comma> testing_text
        IDs, tags = prepro_tag(testing_text_path)
        self._load_model()
        for out in range(IDs.shape[0]):
            # sample_(testing_text_id)_(sample_id).jpg
            noise = numpy.random.normal( 0, 1, ( 5, self.noise_dim))
            gen_imgs = self.generator.predict(numpy.append(noise, tags[i], axis=1))

            # Rescale images 0 ~ 255
            gen_imgs = 127.5 * gen_imgs + 127.5
            gen_imgs = gen_imgs.astype(dtype=numpy.uint8)
            for i in range(1, 6):
                imsave('samples/sample_'+ str(IDs[out]) + '_' + str(i) +'.jpg', skimage.transform.resize(gen_imgs[i], (64, 64)))

    def save_imgs(self, epoch):
        noise = numpy.random.normal( 0, 1, ( 32, self.noise_dim))
        rdm_tags = numpy.zeros( ( 32, self.color_dim), dtype=numpy.float)
        for i in range( 32):
            rdm_tags[i][numpy.random.randint( len(hair_color) )] = 1
            rdm_tags[i][len(hair_color) + numpy.random.randint( len(eye_color) )] = 1

        gen_imgs = self.generator.predict(numpy.append(noise, rdm_tags, axis=1))

        # Rescale images 0 ~ 255
        gen_imgs = 127.5 * gen_imgs + 127.5
        gen_imgs = gen_imgs.astype(dtype=numpy.uint8)
        for i in range(gen_imgs.shape[0]):
            imsave('samples/' + str(epoch) + ' ' + str(hair_color[numpy.nonzero(rdm_tags[i])[0][0]])
                   + ' ' + str(eye_color[numpy.nonzero(rdm_tags[i])[0][1]-len(hair_color)]) + '.jpg', gen_imgs[i])
            # imsave('samples/'+ str(epoch) + '_' + str(i) +'_64.jpg', skimage.transform.resize(gen_imgs[i], (64, 64)))
    

    def _load_model(self):
        self.discriminator = load_model( self.discriminator_path )
        self.generator = load_model( self.generator_path )

if __name__ == '__main__':
    dcgan = DCGAN()
    if len(sys.argv) > 1:
        dcgan.test(sys.argv[1])
    else:
        dcgan.train(batch_size=64, save_interval=50)