#!/usr/local/bin/python3


import pickle
import pprint


descriptions = pickle.load(open('descriptions.p', 'rb'))

pprint.pprint(descriptions)