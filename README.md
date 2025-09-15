# FIO Data Analysis and Machine Learning Project

## Project Overview

This project explores and analyzes datasets generated using the FIO tool (Flexible I/O Tester) to understand disk write performance. The main goal is to predict **latency** based on input parameters such as **WIOPS** and **dispatch queue size (QSize)** using machine learning.

The project includes regression analysis, time series analysis, and predictive modeling with **XGBoost**. All functionalities are integrated into a **command-line interface (CLI)** that allows users to select datasets, choose features, and generate plots or train models.

---

## Running the Software

The project includes a Python virtual environment with all dependencies installed. To run the software, follow these steps:

### 1. Activate the Virtual Environment

Navigate to the root directory of the project folder, then activate the virtual environment:

- **Windows**:

```bash
venv\Scripts\activate
```

### 2. Run the "cli.py" script from the root directory of the project.

```bash
python cli.py
```
