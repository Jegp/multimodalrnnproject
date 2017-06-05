# Multimodal RNNs
A project on using recurrent neural networks (LSTMs) on multimodal data for person recognition

## Prerequisites
To run this project you need to install

* Python 3
* Keras - deep learning framework
* Tensorflow or Theano - machine learning frameworks (required by Keras)
* Matplotlib

To use the hyperparameter tuning, you are required to install Hyperas (Keras + Hyperopt),
which can be found here: https://github.com/maxpumperla/hyperas

All of the above can be installed using pip. I strongly recommend doing this in a virtualenv.

## File layout
We have tree models: one using only audio data, one using only video data and one using both
(dualmodal).

The files containing ``hyper`` is concerned with hyper-parameter optimisation, while the files
``unimodal_audio.py``, ``unimodal_video.py`` and ``dualmodal.py`` contains the optimised
model parameters.

## How to use
To run the already optimized files, simply pull this project and run

    python3 dualmodal.py

This example runs the dualmodal model.
