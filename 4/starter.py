import pickle
import pandas as pd
import numpy as np
import os
import typer

app = typer.Typer()

def read_data(filename):
    categorical = ['PULocationID', 'DOLocationID']
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def process_data(year: int, month: int):
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)

    df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet')

    categorical = ['PULocationID', 'DOLocationID']
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    mean_prediction = np.mean(y_pred)

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

    file_size = os.path.getsize(output_file) / (1024 * 1024)

    return mean_prediction, file_size

@app.command()
def predict(
    year: int = typer.Option(help="Year of the trip data"),
    month: int = typer.Option(help="Month of the trip data")
):
    mean_prediction, file_size = process_data(year, month)
    print(f"Mean predicted duration: {mean_prediction:.2f}")
    print(f"File size: {file_size:.2f} MB")

if __name__ == "__main__":
    app()

# python starter.py --year 2023 --month 4
