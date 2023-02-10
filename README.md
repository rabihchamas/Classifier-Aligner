# Classifier-Aligner
This repository contains an implementation of a deep learning architecture that consists of a classifier and an Aligner. The classifier maps data to classes, and the Aligner tries to match the data to the classifier output.

The Classifier Aligner Network is a deep learning architecture that embodies a game with classifier and aligner agents. The classifier maps a set of input data to class representations, known as class vectors. Meanwhile, the Aligner tries to correlate the shuffled data to the corresponding class vectors.
The architecture operates on the principle that the Aligner can only match the data to the corresponding class vectors if the classifiers' outputs correlate with the input. 
We introduce a loss function using a shuffling technique. After shuffling each of the input data and class vectors randomly in the same way, the Aligner task is to produce class vectors that align with the corresponding shuffled data. We compute the loss as the difference between the Aligner's output and the shuffled class vectors.
In essence, we design a Classifier Aligner Network to achieve unsupervised learning through a collaborative effort between the classifier and the Aligner.
