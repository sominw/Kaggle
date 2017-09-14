from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np

datagen = ImageDataGenerator(
        rotation_range = 40,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        rescale = 1./255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        fill_mode = 'nearest')

img = load_img('/Users/sominwadhwa/Work/data/DogsVsCats/train/Cats/cat.5.jpg')
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

index = 0
for batch in datagen.flow(x, batch_size=1, save_to_dir = 'preview', save_prefix = 'cat', save_format = 'jpeg'):
    index += 1
    if index > 20:
        break

print ("Done")
