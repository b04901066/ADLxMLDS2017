#-*- coding:utf-8 -*-
import os, sys, csv, numpy, pandas, skimage.io, skimage.transform
from scipy.misc import imsave
from keras.models import load_model
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

if __name__ == '__main__':
    # testing_text_id <,> testing_text
    IDs, hair_tags, eyes_tags = prepro_tag( sys.argv[1] )

    model = load_model( './g.h5' )
    noises = [ numpy.load('./noises/1.npy'),
               numpy.load('./noises/2.npy'),
               numpy.load('./noises/3.npy'),
               numpy.load('./noises/4.npy'),
               numpy.load('./noises/5.npy') ]

    for output_count in range( IDs.shape[0] ):
        for i in range(5):
            if numpy.count_nonzero(hair_tags[output_count]) == 0:
                hair_tags[output_count][1] = 1
            if numpy.count_nonzero(eyes_tags[output_count]) == 0:
                eyes_tags[output_count][8] = 1
            gen_img = model.predict( [ noises[i].reshape(1, 100),
                                       hair_tags[output_count].reshape(1, 12),
                                       eyes_tags[output_count].reshape(1, 11) ] )
            # rescale images to 0 ~ 255
            gen_img = 127.5 * gen_img + 127.5
            gen_img = gen_img.astype(dtype=numpy.uint8)
            # sample_(testing_text_id)_(sample_id).jpg
            imsave('samples/extra_sample_' + str(IDs[output_count]) + '_' + str(i+1) + '.jpg',
                   skimage.transform.resize( gen_img[0], (64, 64)) )
