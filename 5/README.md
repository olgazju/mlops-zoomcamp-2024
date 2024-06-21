## What I did:

1. Changed baseline_model_nyc_taxi_data.ipynb that loads green taxi data from 2024-03 as a new production data and applies data transformations on it
2. Then I added  **ColumnQuantileMetric(column_name='fare_amount', quantile=0.5)** and  **ColumnSummaryMetric(column_name='trip_distance')** metrics.
3. I run Report in the loop between 2024-03-01 and 2024-03-30.
4. Then I built Evidently Dashboard with new metrics
<img width="1466" alt="image" src="https://github.com/olgazju/mlops-zoomcamp-2024/assets/14594349/6eb96213-362d-4e23-bc3c-74262678f2a9">
5. Then I made changes in evidently_metrics_calculation.py file adding my new metrics and run it
6. Then I added new graphs in Grafana
<img width="590" alt="image" src="https://github.com/olgazju/mlops-zoomcamp-2024/assets/14594349/578b0d72-7f64-45bc-8e77-e233ce52f209">
