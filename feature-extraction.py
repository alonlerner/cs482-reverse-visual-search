import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import preprocess_input

# load model
base_model = ResNet50(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)

image_size = 224
# initiate output array
features_array = np.zeros((49474,2048))

# loop over all the bounding boxes
for i in range(49474):
    # preprocess image
    img = image.load_img(f'boxes_images/{i}.jpg', target_size=(image_size, image_size))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    # exract features
    features = model.predict(x)
    features = features.reshape(2048,)
    features_array[i,:] = features

# save into npy file
with open('features.npy', 'wb') as f:
    np.save(f, features_array)