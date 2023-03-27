#!/usr/bin/env python
# coding: utf-8

# In[3]:


import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


get_ipython().system('pip install tensorflow')


# In[4]:


(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()


# In[5]:


X_train.shape


# In[6]:


X_test.shape


# In[7]:


y_train[:5]


# In[8]:


y_train = y_train.reshape(-1,)
y_train[:5]


# In[9]:


y_test = y_test.reshape(-1,)


# In[10]:


classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog','horse', 'ship', 'truck'] 


# In[15]:


def plot_sample(X, y, index):
    plt.figure(figsize=(15, 2))
    plt.imshow(X[index])
    plt.xlabel(classes[y[index]])


# In[16]:


plot_sample(X_train, y_train, 5)


# In[20]:


plot_sample(X_train, y_train, 501)


# In[21]:


X_train = X_train / 255.0
X_test = X_test / 255.0


# In[22]:


ann = models.Sequential([
    layers.Flatten(input_shape = (32, 32, 3)),
    layers.Dense(3000, activation = 'relu'),
    layers.Dense(1000, activation = 'relu'),
    layers.Dense(10, activation = 'softmax')
])
ann.compile(optimizer='SGD',
           loss='sparse_categorical_crossentropy',
           metrics=['accuracy'])
ann.fit(X_train, y_train, epochs=5)


# In[23]:


from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
y_pred = ann.predict(X_test)
y_pred_classes = [np.argmax(element) for element in y_pred]
print('classification report: \n', classification_report(y_test, y_pred_classes))


# In[24]:


import seaborn as sns


# In[27]:


plt.figure(figsize = (14, 7))
sns.heatmap(y_pred, annot = True)
plt.ylabel('Truth')
plt.xlabel('Prdiction')
plt.title('Confusion matrix')
plt.show


# In[32]:


cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(32,32,3)),
    layers.MaxPooling2D((2, 2)), 
    
    layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])


# In[34]:


cnn.compile(optimizer='adam',
           loss='sparse_categorical_crossentropy',
           metrics=['accuracy'])


# In[35]:


cnn.fit(X_train, y_train, epochs=10)


# In[36]:


cnn.evaluate(X_test, y_test)


# In[37]:


y_pred = cnn.predict(X_test)
y_pred[:5]


# In[39]:


y_classes = [np.argmax(element) for element in y_pred]
y_classes[:5]


# In[40]:


plot_sample(X_test, y_test, 60)


# In[41]:


plot_sample(X_test, y_test, 400)


# In[43]:


classes[y_classes[60]]


# In[ ]:




