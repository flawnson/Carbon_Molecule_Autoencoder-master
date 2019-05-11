# Project Carbon Coded

## Description
A carbon molecule vanilla autoencoder. The latent layer is comprised of 7 nodes, and is a condensed representation of the learned features from the dataset of 12,000 data points. The latent layer is trained to understand the possible structures carbon is capable of forming. The data preprocessing is comprised of opening the data file, formatting the strings, and appending them into an array. The arrays are then turned into one hot encodings, each encoding representing another molecule. The one hot vectors are then fitted into the model.

## Article
Here's an article I wrote about building this project, as well as how and why it failed
https://towardsdatascience.com/building-a-carbon-molecule-autoencoder-21973e5f88b6
