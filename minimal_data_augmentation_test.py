
from extract_bottleneck_features import extract_Xception

import keras
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential

# don't allow full gpu memory if not required
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True

#set_session(tf.Session(config=config))


from PIL import ImageFile


from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
from glob import glob

from keras.applications.xception import Xception, preprocess_input
from keras.applications.resnet50 import ResNet50

from keras.models import Model

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))



net = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
top_model = Sequential()
dropout_layer = keras.layers.core.Dropout(0.2, input_shape=net.output_shape[1:])
top_model.add(dropout_layer)
top_model.add(GlobalAveragePooling2D())
#top_model.add(GlobalAveragePooling2D(input_shape=net.output_shape[1:]))
top_model.add(Dense(133, activation='softmax'))
my_model = Model(inputs=net.input, outputs=top_model(net.output))

#net = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

## freezing the top level model weight for now, TODO: maybee try to freeze only the firsts 10 layers etc..
for layer in my_model.layers[:-1]:
    layer.trainable = False


# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

# load train, test, and validation datasets
train_files, train_targets = load_dataset('dogImages/train')
valid_files, valid_targets = load_dataset('dogImages/valid')
test_files, test_targets = load_dataset('dogImages/test')

# load list of dog names
dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]

# print statistics about the dataset
print('There are %d total dog categories.' % len(dog_names))
print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.'% len(test_files))


from keras.preprocessing import image
from tqdm import tqdm

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

ImageFile.LOAD_TRUNCATED_IMAGES = True

# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255


# we use a simple dictionary to associate each extract/preprocess function to each network name
# preloading/downloading weight of each network
#extract_Xception(train_tensors)

from keras.preprocessing.image import ImageDataGenerator
# Trying to augment the data:\
augmentation_gen = ImageDataGenerator(
        shear_range=0.4,
        zoom_range=0.2,
        horizontal_flip=True)

augmentation_gen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

#augmentation_gen = ImageDataGenerator()
test_datagen = ImageDataGenerator()


batch_size = 50

#train_generator = get_generator(train_tensors, train_targets, batch_size=batch_size, image_data_generator=augmentation_gen)

train_generator = augmentation_gen.flow(train_tensors, train_targets, batch_size=batch_size)
validation_generator = ImageDataGenerator().flow(valid_tensors, valid_targets, batch_size=batch_size)


#my_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# test_generator = get_generator(batch_size, test_tensors, test_datagen)

if False:
    checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.xception_with_data_augmentation.hdf5', verbose=1,
                                   save_best_only=True)



    my_model.fit_generator(train_generator, steps_per_epoch=len(train_targets)/batch_size,
                           epochs=50, validation_data=validation_generator,
                           validation_steps=len(valid_targets)/batch_size, callbacks=[checkpointer], verbose=1)






test_tensors = paths_to_tensor(test_files).astype('float32')/255
#dog_breed_predictions = [np.argmax(my_model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

# report test accuracy
#test_accuracy = 100*np.sum(np.array(dog_breed_predictions)==np.argmax(test_targets, axis=1))/len(dog_breed_predictions)
#print('Test accuracy: %.4f%%' % test_accuracy)

# TODO: maintenant que le modele et le dernier layer a convergé.. on pourrait retrained.. en utilisant un otimizer


#TODO: laurent uitliser autre chose que rmsprop ? ou mieux comprendre.. et avoir un truc qui est plus petit

# reloading:


#for layer in my_model.layers[:-1]:
#    layer.trainable = True

print("Relearning part of the network..")
# we made only the first 20 layers not trainable, the rest is trainablabe
for num, layer in enumerate(my_model.layers):
    if num < 20:
        layer.trainable = False
    else:
        layer.trainable = True

my_model.load_weights(filepath='saved_models/weights.best.xception_with_data_augmentation.hdf5')
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.xception_with_data_augmentation_after_relearning.hdf5', verbose=1,
                               save_best_only=True)
def get_accuracy(my_model):
    dog_breed_predictions = [np.argmax(my_model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]
    test_accuracy = 100*np.sum(np.array(dog_breed_predictions)==np.argmax(test_targets, axis=1))/len(dog_breed_predictions)
    print('Test accuracy: %.4f%%' % test_accuracy)
    return test_accuracy

optimizer = keras.optimizers.rmsprop(lr=0.0001)
my_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
print("Start")
my_model.fit_generator(train_generator, steps_per_epoch=len(train_targets)/batch_size, epochs=10, validation_data=validation_generator, validation_steps=len(valid_targets)/batch_size, verbose=1, callbacks=[checkpointer])
get_accuracy(my_model)

print("Again")
my_model.fit_generator(train_generator, steps_per_epoch=len(train_targets)/batch_size, epochs=10, validation_data=validation_generator, validation_steps=len(valid_targets)/batch_size, verbose=1, callbacks=[checkpointer])
get_accuracy(my_model)

# TODO: je ne comprends pas pourquoi on repasse a un val_acc vachement faible la.. TODO: reesayer en laissant les premiers layer freezed.. si le load fonctionne bien on devrait avoir un val_acc sur la premiere epoch a 0.73 et 0.10
# j'ai vraiment l'impression que ca vient de rmsprop.. prendre un sgd classique avec un petit learning rate fera sans doute le job



my_model.load_weights(filepath='saved_models/weights.best.xception_with_data_augmentation.hdf5')
get_accuracy(my_model)

if False:
    optimizer = keras.optimizers.rmsprop(lr=0.0001)
    my_model.load_weights(filepath='saved_models/weights.best.xception_with_data_augmentation.hdf5')

    checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.xception_with_data_augmentation_final_aprem.hdf5', verbose=1,
                                   save_best_only=True)

    my_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    my_model.fit_generator(train_generator, steps_per_epoch=len(train_targets)/batch_size, epochs=500, validation_data=validation_generator, validation_steps=len(valid_targets)/batch_size, verbose=1, callbacks=[checkpointer])
    get_accuracy(my_model)
    import IPython
    IPython.embed()

    my_model.fit_generator(train_generator, steps_per_epoch=len(train_targets)/batch_size, epochs=1, validation_data=validation_generator, validation_steps=len(valid_targets)/batch_size, verbose=1)
    get_accuracy(my_model)
    # we have 83.0144 % with this model, that's better than the 80.622 % previously without full retraining, but without data augmentation we have Test accuracy for model based on Xception: 83.6124%
    # thus the dataagumentation strategies didn't seems to be really impressive here




my_model.load_weights(filepath='saved_models/weights.best.xception_with_data_augmentation_final_aprem.hdf5')
get_accuracy(my_model)


#test_targets = np.argmax(test_targets, axis=1)
dog_breed_predictions = [np.argmax(my_model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

targets = np.argmax(test_targets, axis=1)
mat = confusion_matrix(targets, dog_breed_predictions)

plt.matshow(mat, cmap='coolwarm', interpolation=None)
