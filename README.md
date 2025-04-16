# IoT Energy Management System For Residential Buildings

An end-to-end solution for simulating IoT energy sensors, securely publishing sensor data to Azure IoT Hub, storing and analyzing historical data, and applying predictive models to optimize energy usage in residential buildings. This system demonstrates how to integrate ARIMA and LSTM forecasting models with smart energy controls and how to route decision data into Azure Blob Storage as CSV files

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Folder Structure](#folder-structure)
4. [Installation & Setup](#installation--setup)
5. [Usage](#usage)
6. [Phases](#phases)
   - [Phase 1: IoT Sensor Simulation & Data Publishing](#phase-1-iot-sensor-simulation--data-publishing)
   - [Phase 2: Data Collection & Historical Data Management](#phase-2-data-collection--historical-data-management)
   - [Phase 3: Predictive Analytics & Forecasting Engine](#phase-3-predictive-analytics--forecasting-engine)
   - [Phase 4: Integration & Energy Demand Optimization](#phase-4-integration--energy-demand-optimization)
7. [License](#license)
8. [Contributing](#contributing)
9. [Contact](#contact)

---

## Overview

This repository showcases a **Smart Energy Management** pipeline designed for residential buildings. It starts with simulated IoT sensors (smart plug, temperature/humidity, and occupancy sensors) sending data to Azure IoT Hub, collects and stores the data, then applies forecasting models (ARIMA & LSTM) to predict energy usage. The predictions are used to make real-time energy distribution decisions, which are then pushed back to the cloud—specifically, written as CSV files into Azure Blob Storage for archival and further analysis.

---

## Features

- **IoT Sensor Simulation**: Randomized sensor data for power usage, temperature/humidity, and occupancy.
- **Secure Data Publishing**: MQTT with SAS token authentication to Azure IoT Hub.
- **Data Collection**: Ongoing logging of sensor readings into a local CSV file for historical reference.
- **Predictive Analytics**:
  - ARIMA with automated grid search for parameter tuning.
  - LSTM with a stacked architecture and dropout layers.
- **Comparison & Diagnostics**: Visualized comparisons, error metrics (MSE, MAE), and residual analyses.
- **Integration**: Real-time decision-making scripts that push energy optimization data (e.g., directly to Azure Blob Storage as CSV files).

---

## Folder Structure

```bash
IoT-Energy-Management-System-For-Residential-Buildings/
├── sensors/
├── publishers/
├── utils/
├── analytics/
├── data/
├── main.py
├── energy_optimizer.py
├── LICENSE
├── requirements.txt
├── .env
└── README.md
```

---

## Installation & Setup

1. **Clone the Repository**

```bash
git clone https://github.com/YourUsername/IoT-Energy-Management-System-For-Residential-Buildings.git
cd IoT-Energy-Management-System-For-Residential-Buildings
```

2. **Create & Activate Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

4. **Configure Environment Variables**
- Copy `.env.example` to `.env`.
- Fill in Azure IoT Hub and storage credentials.
- Set `PUBLISH_INTERVAL` as needed.

---

## Usage

### 1. Start Sensor Data Publishing

```bash
python main.py
```

Or directly:

```bash
python publishers/mqtt_azure_publisher.py
```

### 2. Collect Data Locally
- Written to `data/sensor_data.csv`.

### 3. Forecasting with ARIMA & LSTM

```bash
python analytics/forecast_arima.py --steps 15 --look_back 24 --epochs 50 --batch_size 32
```

### 4. Energy Optimization & Integration
- `energy_optimizer.py` sends decisions every 15s to IoT Hub.
- ASA job writes those decisions to Azure Blob Storage as CSV.

---

## Phases

### Phase 1: IoT Sensor Simulation & Data Publishing
- Simulated sensors send data securely to Azure IoT Hub.

### Phase 2: Data Collection & Historical Data Management
- Logged to local CSV for storage.

### Phase 3: Predictive Analytics & Forecasting Engine
- ARIMA and LSTM models forecast future energy usage.
- Outputs include CSVs, PNGs, and metrics.

### Phase 4: Integration & Energy Demand Optimization
- Decisions are pushed to Azure and stored as CSV via ASA.
- Utilized by external dashboards or control interfaces.

---

## License

Licensed under the [MIT License](LICENSE).

---

## Contributing

1. Fork this repo  
2. Create a branch `feature/xyz`  
3. Commit your changes  
4. Push and open a Pull Request

---

## Contact

- Open a GitHub issue  
- Submit a Pull Request  

---

Happy hacking!
