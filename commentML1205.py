#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf


# In[2]:


train_df = pd.read_csv('train1205.csv')
test_df = pd.read_csv('val1205.csv')
print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")


# In[3]:


train_df = train_df.dropna(subset=["review_text","taget"])
test_df = test_df.dropna(subset=["review_text"])


# In[4]:


train_df = train_df.fillna(value={"review_id":"_nan_"})
test_df = test_df.fillna(value={"review_id":"_nan_"})


# In[5]:


print("Shapes after NaN valus hadled")
print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")


# In[6]:


train_df, val_df = train_test_split(train_df,
                                    test_size=0.1,
                                    random_state=2000)


# In[7]:


training_sentences = list(train_df["review_text"].values)
val_sentences = val_df["review_text"].values
test_sentences = test_df["review_text"].values


# In[8]:


embed_size = 300
max_features = 50000
maxlen = 100
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_features, oov_token="<oov>")
tokenizer.fit_on_texts(training_sentences)

X_train = tokenizer.texts_to_sequences(training_sentences)
X_val = tokenizer.texts_to_sequences(val_sentences)
X_test = tokenizer.texts_to_sequences(test_sentences)


# In[9]:


X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train,
                                                        maxlen=maxlen,
                                                        padding="post",
                                                        truncating="post")
X_val = tf.keras.preprocessing.sequence.pad_sequences(X_val,
                                                        maxlen=maxlen,
                                                        padding="post",
                                                        truncating="post")
X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test,
                                                        maxlen=maxlen,
                                                        padding="post",
                                                        truncating="post")


# In[10]:


y_train = train_df["taget"].values
y_val = val_df["taget"].values


# In[11]:


inputs = tf.keras.layers.Input(shape=(maxlen,))
x = tf.keras.layers.Embedding(max_features, embed_size)(inputs)
x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True))(x)
x = tf.keras.layers.GlobalMaxPool1D()(x)
x = tf.keras.layers.Dense(16, activation="relu")(x)
x = tf.keras.layers.Dropout(0.1)(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()


# In[12]:


import time
start = time.time()


# In[13]:


hist = model.fit(X_train, y_train, batch_size=4096, epochs=1, validation_data=(X_val, y_val))
y_pred = model.predict(X_test, batch_size=1024)
print(time.time()-start)


# In[14]:


y_te = (y_pred[:,0] > 0.5).astype(np.int_)
submit_df = pd.DataFrame({"review_id": test_df["review_id"], "prediction": y_te})
submit_df.to_csv("test_output1205.csv", index=False)


# In[15]:


dftest = pd.read_csv('test_output1205.csv')
dftest.head(20)


# In[ ]:




