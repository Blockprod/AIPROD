# Prometheus + Grafana Integration for AIPROD V33

## Overview

This project now uses Prometheus and Grafana for monitoring and metrics collection.

## How to Use

### 1. Start the Monitoring Stack

Run the following command to start the API, Prometheus, and Grafana:

    docker-compose up --build

- The API will be available at: http://localhost:8000
- Prometheus UI: http://localhost:9090
- Grafana UI: http://localhost:3000 (login: admin / admin)

### 2. Metrics Endpoint

The FastAPI app exposes metrics at:

    http://localhost:8000/metrics

Prometheus is configured to scrape this endpoint automatically.

### 3. Grafana Dashboard

- Open Grafana at http://localhost:3000
- Add Prometheus as a data source (URL: http://prometheus:9090)
- Import or create dashboards to visualize API metrics.

## Configuration Files

- `config/prometheus.yml`: Prometheus scrape config
- `config/grafana/`: Grafana persistent data

## Example Prometheus Query

To see request count:

    http_requests_total

## Notes

All legacy monitoring code, dependencies, and configs have been removed.

- For custom metrics, use the Prometheus FastAPI Instrumentator in your code.

---

For more details, see the FastAPI Instrumentator docs: https://github.com/trallard/prometheus-fastapi-instrumentator
