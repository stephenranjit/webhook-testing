#!/usr/local/bin/python3

'''
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub

from tensorflow.python.keras.utils import losses_utils

import pprint

from sklearn.preprocessing import MultiLabelBinarizer
'''

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import json
import pickle
import urllib
import pprint

from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.python.keras.utils import losses_utils

print(tf.__version__)



data = pd.read_csv('herbbenefits-training.csv')


#descriptions = data['overview']
#genres = data['genres']

urllib.request.urlretrieve('https://storage.googleapis.com/bq-imports/descriptions.p', 'descriptions.p')
urllib.request.urlretrieve('https://storage.googleapis.com/bq-imports/genres.p', 'genres.p')

descriptions = pickle.load(open('descriptions.p', 'rb'))
genres = pickle.load(open('genres.p', 'rb'))


print('+++++++++++++++++')

print(descriptions)

print('+++++++++++++++++')

print(genres)


print('+++++++++++++++++')


#f = open('labels2', encoding='utf-8')

#top_genres = f.readlines()

#print(top_genres)

train_size = int(len(descriptions) * .8)

train_descriptions = descriptions[:train_size].astype('str')
train_genres = genres[:train_size]

test_descriptions = descriptions[train_size:].astype('str')
test_genres = genres[train_size:]



encoder = MultiLabelBinarizer()
encoder.fit_transform(train_genres)
train_encoded = encoder.transform(train_genres)
test_encoded = encoder.transform(test_genres)
num_classes = len(encoder.classes_)

# Print all possible genres and the labels for the first movie in our training dataset
print(encoder.classes_)
print(train_encoded[0])

description_embeddings = hub.text_embedding_column("movie_descriptions", module_spec="https://tfhub.dev/google/universal-sentence-encoder/2", trainable=False)

multi_label_head = tf.estimator.MultiLabelHead(
    num_classes,
    loss_reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE
)
'''
multi_label_head = tf.estimator.MultiLabelHead(
    num_classes,
    #loss_reduction=tf.compat.v1.losses.Reduction.SUM_OVER_BATCH_SIZE
    loss_reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE

)'''

features = {
  "movie_descriptions": np.array(train_descriptions).astype(np.str)
}
labels = np.array(train_encoded).astype(np.int32)
train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(features, labels, shuffle=True, batch_size=8, num_epochs=75)
estimator = tf.estimator.DNNEstimator(
    head=multi_label_head,
    hidden_units=[64,10],
    feature_columns=[description_embeddings])


estimator.train(input_fn=train_input_fn)

eval_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn({"movie_descriptions": np.array(test_descriptions).astype(np.str)}, test_encoded.astype(np.int32), shuffle=False)

estimator.evaluate(input_fn=eval_input_fn)

'''raw_test = [
    "The bark decoction of the tree is administered in cases of malaria, while the leaves are used in jaundice.",   #malaria,jaundice
    "Marmin, a coumarin isolated from the roots has anti-inflammatory  properties, which makes Bael fruit ideal for treating inflammation." #inflammation
]'''

raw_test = [
    "An examination of our dietary choices and the food we put in our bodies. Based on Jonathan Safran Foer's memoir.", # Documentary
    "After escaping an attack by what he claims was a 70-foot shark, Jonas Taylor must confront his fears to save those trapped in a sunken submersible.", # Action, Adventure
    "A teenager tries to survive the last week of her disastrous eighth-grade year before leaving to start high school.", # Comedy
]


predict_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn({"movie_descriptions": np.array(raw_test).astype(np.str)}, shuffle=False)

results = estimator.predict(predict_input_fn)


# Display predictions
for movie_genres in results:
  top_2 = movie_genres['probabilities'].argsort()[-2:][::-1]
  print('~~~~~~~~~~~~~~~~')
  pprint.pprint(top_2)
  print('~~~~~~~~~~~~~~~~')
  for genre in top_2:
    text_genre = encoder.classes_[genre]
    print(text_genre + ': ' + str(round(movie_genres['probabilities'][genre] * 100, 2)) + '%')
  print('')