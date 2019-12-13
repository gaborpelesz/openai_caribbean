import warnings
warnings.filterwarnings('ignore')

import numpy as np
import os

from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout, Reshape
from tensorflow.python.keras.layers import Dense
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint 

from sklearn.utils import class_weight
# In[22]:


model = Sequential()

model.add(ResNet50(include_top = False, pooling = 'avg', weights = 'imagenet'))

model.add(Dense(1024, activation="relu"))
model.add(Dense(512, activation="relu"))
model.add(Dense(256, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(5, activation = 'softmax'))

model.layers[0].trainable = False


# In[23]:


model.summary()


# In[24]:


model.compile(optimizer='adam', metrics=['accuracy'],loss='categorical_crossentropy')
es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=10)
mc = ModelCheckpoint('model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

# In[25]:


base_dir = 'training_data'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')


# In[26]:


image_size = (224,224)
batch_size = 10
seed = 1

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=image_size, batch_size=batch_size, class_mode='categorical', seed=seed, color_mode="rgb")
validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=image_size, batch_size=batch_size, class_mode='categorical', seed=seed, color_mode="rgb")

class_weights = class_weight.compute_class_weight(
           'balanced',
            np.unique(train_generator.classes), 
            train_generator.classes)


# In[ ]:


model.fit_generator(train_generator, validation_data=validation_generator, epochs=50, class_weight=class_weights, callbacks=[es, mc])
model.save('models/karibi.h5')

# In[ ]:




