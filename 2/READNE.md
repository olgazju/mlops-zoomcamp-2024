## How to install mlflow

```
pip install mlflow
```

## How to run mlflow server

``` 
mkdir -p mlflow_server/sqlite
mkdir -p mlflow_server/artifacts
mlflow server \
    --backend-store-uri sqlite:///mlflow_server/sqlite/mlflow.db \
    --default-artifact-root ./mlflow_server/artifacts \
    --host 0.0.0.0 \
    --port 5000
```