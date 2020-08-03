#!/usr/local/bin/python3

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub

import pprint

from sklearn.preprocessing import MultiLabelBinarizer

data = pd.read_csv('herbbenefits-training.csv')

descriptions = data['sentence']
genres = data['benefits']


#print(sentences)

f = open('labels2', 'r')

top_genres = f.readlines()

#print(top_genres)

train_size = int(len(descriptions) * .8)

train_descriptions = descriptions[:train_size]
train_genres = genres[:train_size]

test_descriptions = descriptions[train_size:]
test_genres = genres[train_size:]

description_embeddings = hub.text_embedding_column(
  "movie_descriptions", 
  module_spec="https://tfhub.dev/google/universal-sentence-encoder/2"
)


encoder = MultiLabelBinarizer()
encoder.fit_transform(train_genres)
train_encoded = encoder.transform(train_genres)
test_encoded = encoder.transform(test_genres)
num_classes = len(encoder.classes_)

multi_label_head = tf.contrib.estimator.multi_label_head(
    num_classes,
    loss_reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE
)

estimator = tf.contrib.estimator.DNNEstimator(
    head=multi_label_head,
    hidden_units=[64,10],
    feature_columns=[description_embeddings]
)

# Format our data for the numpy_input_fn
features = {
  "descriptions": np.array(train_descriptions)
}
labels = np.array(train_encoded)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    features, 
    labels, 
    shuffle=True, 
    batch_size=32, 
    num_epochs=20
)