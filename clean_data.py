import pandas as pd
import numpy as np


def show_val_counts(dataset, name):
    print(dataset[name].value_counts())
