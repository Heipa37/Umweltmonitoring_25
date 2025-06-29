FROM python:3.12


# Set working directory for prefect and dbt.
WORKDIR /app

# Copy requirements and install them.
COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt --no-cache-dir

COPY senseboxAPI.py .
COPY db_management.py .
COPY ML_forecast.py .
COPY DASH.py .

EXPOSE 8050

CMD ["python", "app.py"]