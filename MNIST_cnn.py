import numpy as np
import pandas as pd
import keras

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable

#load the data
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

X_train