# FIO Data Analysis and Machine Learning Project

## Project Overview

This project analyzes datasets generated with the **FIO (Flexible I/O Tester)** tool to study disk I/O performance, particularly focusing on **latency** under different workloads. The main goal is to explore the relationships between variables such as **WIOPS** (write I/O operations per second) and **dispatch queue size (QSize)**, and to predict latency using machine learning techniques.

The application provides a **command-line interface (CLI)** that allows users to:

- Select one or more datasets for analysis
- Perform **2D or 3D polynomial regression** to study dependencies between variables
- Plot **time series** for individual variables or combined datasets
- Train and evaluate **XGBoost regression models** for latency prediction

## The system is designed to be flexible and scalable, automatically handling multiple datasets, removing outliers, and saving plots and trained models in timestamped folders for easy tracking. This makes it useful for both exploratory data analysis and predictive modeling of disk performance.

## Running the Software

The project includes a requirements file with all the necessary dependencies. To run the software, follow these steps:

### 1. Activate the Virtual Environment

Navigate to the root directory of the project folder, then activate the virtual environment:

- **Windows**:

```bash
venv\Scripts\activate
```

- **Linux**:

```bash
source venv/bin/activate
```

### 2. Install de dependencies from the given file.

```bash
pip install -r requirements.txt
```

### 3. Run the "index.py" as a module script from the root directory of the project.

- **Windows**:

```bash
python -m index
```

- **Linux**:

```bash
python3 -m index
```
