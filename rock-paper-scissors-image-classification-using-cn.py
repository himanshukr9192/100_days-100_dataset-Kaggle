#!/usr/bin/env python
# coding: utf-8

# # Libraries

# In[24]:


import os
import glob

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout


# # Dataset

# In[25]:


# list the files in the folder
#!ls ../input/rock-paper-scissors-dataset/Rock-Paper-Scissors


# In[26]:


# list the files in the folder
#!ls ../input/rock-paper-scissors-dataset/Rock-Paper-Scissors/test


# In[27]:


base_dir = r'C:\Users\himan\Rock Paper Scissors\Rock-Paper-Scissors'

train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'test')
test_dir = os.path.join(base_dir, 'validation')

paper_dir = os.path.join(train_dir, 'paper')
rock_dir = os.path.join(train_dir, 'rock')
scissors_dir = os.path.join(train_dir, 'scissors')

paper_imgs = os.listdir(paper_dir)
rock_imgs = os.listdir(rock_dir)
scissors_imgs = os.listdir(scissors_dir)


# In[28]:


plt.figure(figsize=(20, 4))
for i, img_path in enumerate(paper_imgs[:5]):
    sp = plt.subplot(1, 5, i+1)
    img = mpimg.imread(os.path.join(paper_dir, img_path))
    plt.imshow(img)
plt.show()


# In[29]:


plt.figure(figsize=(20, 4))
for i, img_path in enumerate(rock_imgs[:5]):
    sp = plt.subplot(1, 5, i+1)
    img = mpimg.imread(os.path.join(rock_dir, img_path))
    plt.imshow(img)
plt.show()


# In[30]:


plt.figure(figsize=(20, 4))
for i, img_path in enumerate(scissors_imgs[:5]):
    sp = plt.subplot(1, 5, i+1)
    img = mpimg.imread(os.path.join(scissors_dir, img_path))
    plt.imshow(img)
plt.show()


# In[31]:


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2)
validation_datagen = ImageDataGenerator(rescale=1.0/255)


# In[32]:


BATCH_SIZE = 32
TARGET_SIZE = 64
EPOCHS = 10


# In[33]:


train_generator = train_datagen.flow_from_directory(train_dir, 
                                                    target_size=(TARGET_SIZE, TARGET_SIZE), 
                                                    batch_size=BATCH_SIZE, 
                                                    shuffle=True,
                                                    class_mode='categorical')

val_generator = validation_datagen.flow_from_directory(val_dir,
                                                       target_size=(TARGET_SIZE, TARGET_SIZE), 
                                                       batch_size=BATCH_SIZE, 
                                                       shuffle=True,
                                                       class_mode='categorical')


# In[34]:


model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='sigmoid'))


# In[35]:


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[36]:


model.summary()


# In[37]:


get_ipython().run_line_magic('pinfo', 'model.fit_generator')


# In[38]:


history = model.fit_generator(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    verbose=1)


# In[39]:


plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.savefig('./foo.png')
plt.show()


# In[40]:


from tensorflow.keras.models import load_model

model.save('rock_paper_scissors_cmp.h5')


# In[ ]:





# In[46]:


import gradio 
import requests
import urllib
import tensorflow as tf
from PIL import Image




model=tf.keras.models.load_model(r'C:\Users\himan\Rock Paper Scissors\rock_paper_scissors_cmp.h5')
labels=['Paper','Rock','Scissors']

def classify_image(inp):
    # Resize the input image to (64, 64)
    inp = Image.fromarray(inp)
    inp = inp.resize((64, 64))
    inp = np.array(inp)

    # Preprocess the input image
    inp = inp[None, ...]
    inp = tf.keras.applications.inception_v3.preprocess_input(inp)

    # Make a prediction using the model
    prediction = model.predict(inp).flatten()
    return {labels[i]: float(prediction[i]) for i in range(no_classes)}


import gradio
import numpy as np
from PIL import Image
import tensorflow as tf

def classify_image(inp):
    if inp is None:
        # Handle the case where inp is None
        return {}
    # Resize the input image to (64, 64)
    inp = Image.fromarray(inp)
    inp = inp.resize((64, 64))
    inp = np.array(inp)

    # Preprocess the input image
    inp = inp[None, ...]
    inp = tf.keras.applications.inception_v3.preprocess_input(inp)

    # Make a prediction using the model
    prediction = model.predict(inp).flatten()
    no_classes = 3  # Set this to the number of classes in your model's output
    return {labels[i]: float(prediction[i]) for i in range(no_classes)}





# In[47]:


image = gradio.components.Image(shape=(224, 224))
label = gradio.components.Label(num_top_classes=3)

gradio.Interface(fn=classify_image, inputs=image, outputs=label, interpretation="default").launch(share=True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




