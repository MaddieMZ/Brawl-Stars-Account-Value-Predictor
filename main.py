import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OrdinalEncoder as oe

def load_raw_data():
    return pd.read_csv('data/ejemplo.csv')

print("Ej" + "\n" + str(load_raw_data()))
