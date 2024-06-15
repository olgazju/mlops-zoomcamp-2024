#!/usr/bin/env python
# coding: utf-8

get_ipython().system('pip freeze | grep scikit-learn')


get_ipython().system('python -V')


import pickle
import pandas as pd


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


year = 2023
month = 3
df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet')


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


import numpy as np
standard_deviation = np.std(y_pred)
print(f"Standard Deviation of the predicted duration: {standard_deviation}")


df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

df_result = pd.DataFrame({
    'ride_id': df['ride_id'],
    'prediction': y_pred
})

output_file = 'results.parquet'
df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)

import os
file_size = os.path.getsize(output_file)
print(f"File size: {file_size / (1024 * 1024):.2f} MB")


get_ipython().system('jupyter nbconvert --to script starter.ipynb')

