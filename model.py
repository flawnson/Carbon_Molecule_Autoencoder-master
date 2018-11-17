import keras as k
from keras.models import Sequential, Model
from keras.layers import Activation, BatchNormalization, Input, Dense
from keras.layers.core import Dense
from keras.optimizers import Adam, SGD
from keras.metrics import categorical_crossentropy
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors, Draw
from rdkit.Chem.Draw import IPythonConsole
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from numpy import argmax
import tensorflow as tf
import math
import re
import string
from collections import OrderedDict

#Formating unlabeled SMILE strings and putting them into an array
formated_SMILE_array = []

with open('lets_get_these_carbon_SMILES.txt') as my_file:
    SMILES_array = my_file.readlines()
    for SMILE in SMILES_array:
        head, sep, tail = SMILE.partition('\t')
        SMILE = tail
        formated_SMILE_array.append(SMILE)

#Normalizing data by adding spaces at the end of each string and putting them into an array (each string is 800 characters in total, to account for very large SMILE strings)
spaced_SMILES = []

for SMILE in formated_SMILE_array:
    width = 800 - len(SMILE)
    padded_SMILE = SMILE.ljust(width)
    spaced_SMILES.append(padded_SMILE)

#Extracting the SMILES Grammer alphabet and all possible characters
unique_characters = []

for spaced_SMILE in spaced_SMILES:
    possible_characters = "".join(OrderedDict.fromkeys(spaced_SMILE).keys())
    unique_characters.append(possible_characters)

characters = "".join(unique_characters)

alphabet = "".join(set(characters))

#Building one hot encoder
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))

integer_encoded_list = []

for spaced_SMILE in spaced_SMILES:
    for char in spaced_SMILE:
        integer_encoded = char_to_int[char]
        integer_encoded_list.append(integer_encoded)

onehot_encoded = list()
for value in integer_encoded_list:
	letter = [0 for _ in range(len(alphabet))]
	letter[value] = 1
	onehot_encoded.append(letter)

#Building the deep autoencoder model (Input )
input_array = Input(shape=(63,))
hidden_1 = Dense(32, activation='relu')(input_array)
hidden_2 = Dense(14, activation='relu')(hidden_2)
code = Dense(7, activation='relu')(hidden_1)
hidden_3 = Dense(14, activation='relu')(code)
hidden_4 = Dense(32, activation='relu')(hidden_3)
output_array = Dense(63, activation='sigmoid')(hidden_4)

#Compiling and fitting the carbon molecule data
autoencoder = Model(input_array, output_array)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(np.array(onehot_encoded), np.array(onehot_encoded), batch_size=2, epochs=3617)