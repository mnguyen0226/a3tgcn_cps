# Reference: https://github.com/lehaifeng/T-GCN/blob/master/T-GCN/T-GCN-TensorFlow/main.py

import pickle as plk
import tensorflow as tf
import pandas as pd
import numpy as np
import math
import os
import numpy.linalg as la
from utils import preprocess_data
from utils import load_clean_scada_data
from models import TGCNCell
from utils import plot_error
from utils import plot_result
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import time


def main():
    print("Start the training process")


if __name__ == "__main__":
    main()
