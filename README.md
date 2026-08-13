# Weather Data Visualizer

Analyzes real-world weather data from a CSV file and generates graphical insights. The program performs data cleaning, statistical analysis using NumPy/Pandas, and visualization using Matplotlib.

## Features
- Load and clean raw CSV weather data
- Handle missing values (drop/interpolate/fill strategy)
- Generate summary statistics
- Create multiple visualizations:
  - Line plot — daily temperature trends
  - Bar chart — monthly rainfall
  - Scatter plot — humidity vs. temperature
  - Combined plot with multiple subplots
- Export cleaned dataset and plots

## Concepts Used
- Pandas for data cleaning and manipulation
- NumPy for statistical calculations
- Matplotlib for visualization
- CSV handling
- Data grouping and aggregation

## How to Run
```bash
pip install -r requirements.txt
python weather_analyzer/main.py
```
_(Confirm the actual entry-point filename inside weather_analyzer/ and update the command above if different.)_

## Output
Cleaned dataset and generated charts (temperature trend, rainfall, humidity vs. temperature) saved to the project directory.
