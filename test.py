import csv
from my_spherical_km import SPKM
import numpy as np

df = []

with open('kmTestInput.csv') as f:
    reader = csv.reader(f, delimiter = ',')
    for row in reader:
        df.append(row)

df = np.array(df)

SPKM(df, 2)