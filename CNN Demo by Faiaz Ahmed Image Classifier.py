#!/usr/bin/env python
# coding: utf-8

# # Convolutional Neural Network

# ### Importing the libraries

# In[48]:


import tensorflow as tf

import os

from keras.preprocessing.image import ImageDataGenerator

from PIL import Image


# ## Part 1 - Data Preprocessing

# ### Preprocessing the Training set

# In[49]:


train_datagen = ImageDataGenerator(
    
    rescale = 1./255,
    
    shear_range = 0.2,
    
    zoom_range = 0.2,
    
    horizontal_flip = True
)

training_set = train_datagen.flow_from_directory(
    
    r'C:\Users\faiaz\Desktop\UpSkill Resources\Section 40 - Convolutional Neural Networks (CNN)\dataset\training_set',
    
    target_size = (64,64),
    
    batch_size = 32,
    
    class_mode = 'binary'
    
)


# ### Preprocessing the Test set

# In[50]:


test_datagen = ImageDataGenerator(rescale = 1./255)

test_set = test_datagen.flow_from_directory(
    
    r'C:\Users\faiaz\Desktop\UpSkill Resources\Section 40 - Convolutional Neural Networks (CNN)\dataset\test_set',
    
    target_size = (64,64),
    
    batch_size = 32,
    
    class_mode = 'binary'
    
)


# ## Part 2 - Building the CNN

# ### Initialising the CNN

# In[51]:


cnn = tf.keras.models.Sequential()


# ### Step 1 - Convolution

# In[52]:


cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3,activation = 'relu', input_shape = [64,64,3]))


# ### Step 2 - Pooling

# In[53]:


cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2,strides = 2))


# ### Adding a second convolutional layer

# In[54]:


cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu'))

cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))


# ### Step 3 - Flattening

# In[55]:


cnn.add(tf.keras.layers.Flatten())


# ### Step 4 - Full Connection

# In[56]:


cnn.add(tf.keras.layers.Dense(units = 128, activation = 'relu'))


# ### Step 5 - Output Layer

# In[57]:


cnn.add(tf.keras.layers.Dense(units = 1, activation='sigmoid'))


# ## Part 3 - Training the CNN

# ### Compiling the CNN

# In[58]:


cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# ### Training the CNN on the Training set and evaluating it on the Test set

# In[59]:


cnn.fit(x = training_set, validation_data = test_set, epochs = 25)


# ## Part 4 - Making a single prediction

# In[64]:


import numpy as np

from tensorflow.keras.preprocessing.image import load_img, img_to_array

test_image = image.load_img(r'C:\Users\faiaz\Desktop\UpSkill Resources\Section 40 - Convolutional Neural Networks (CNN)\dataset\single_prediction\cat_or_dog_3.jpg', target_size = (64,64))

test_image = image.img_to_array(test_image)

test_image = np.expand_dims(test_image, axis = 0)

result = cnn.predict(test_image/255.0)

training_set.class_indices

if result[0][0] > 0.5:
    
    prediction = 'dog'
    
else:
     
    prediction = 'cat'


# In[67]:


print(prediction)


# In[ ]:





# In[ ]:




