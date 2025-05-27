import pickle
import pandas as pd

def load_obj(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def load_data(filepath, date_cutoff):
    df = load_obj(filepath)
    df = df[df.dt <= date_cutoff]
    return df


