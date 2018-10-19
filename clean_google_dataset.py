import time
from datetime import datetime

import pandas as pd
import numpy as np

multipliers = {'k': 10e2, 'M': 10e5}
millions_converter = lambda x: int(float(x[:-1])*multipliers[x[-1]]) #not sure abour 10e5, not 10e6
plus_remover = lambda x: int(float(x[:-1].replace(',', '')))
type_converter = lambda x: bool(0 if x == 'Free' or x == 0 else 1)
price_converter = lambda x: float(x if type(x) == int else x.replace('$', ''))
date_converter = lambda x: int(time.mktime(datetime.strptime(x, '%B %d, %Y').timetuple()))


def convert_columns_to_int(value):
    if type(value) == int:
        value = value
    elif "M" in value or "k" in value:
        value = millions_converter(value)
    elif "Varies with device" in value:
        value = 0
    elif "+" in value:
        value = plus_remover(value)
    else:
        value = int(value)
    return value


def convert_types_to_int(value, types):
    return types.index(value)


def prepare_features(dataset, feature_names):
    dataset[dataset.keys()[1]] = dataset[dataset.keys()[1]].replace(np.NaN, -1)
    data_list = dataset.values.tolist()

    categories = list(set([row[0] for row in data_list]))
    content_ratings = list(set([row[5] for row in data_list]))
    genres = list(set([row[6] for row in data_list]))

    for row in data_list:
        row[0] = convert_types_to_int(row[0], categories)
        row[2] = convert_columns_to_int(row[2])
        row[3] = convert_columns_to_int(row[3])
        row[4] = price_converter(row[4])
        row[5] = convert_types_to_int(row[5], content_ratings)
        row[6] = convert_types_to_int(row[6], genres)
        row[7] = date_converter(row[7])

    return pd.DataFrame(data_list, columns=feature_names)
