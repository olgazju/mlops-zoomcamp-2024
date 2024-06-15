FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

WORKDIR /app

COPY starter.py .

RUN pip install typer==0.12.3 pandas==2.2.2 pyarrow==15.0.2 scikit-learn==1.5.0

# Command to run the script with Typer CLI
ENTRYPOINT ["python", "starter.py"]

# docker build -t taxi-predictor .
# docker run taxi-predictor --year 2023 --month 5