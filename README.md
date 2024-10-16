# Crop Yield Prediction Using MLP-GRU Model

This project is focused on predicting crop yield based on factors like area, rainfall, fertilizer usage, pesticide usage, and season using a hybrid neural network model with MLP and GRU architectures. The dataset contains historical agricultural data, including various crop-related attributes, across multiple states and years in India.

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)

## Installation

To run this project, install the following libraries (if not already installed):

```bash
pip install numpy pandas matplotlib seaborn tensorflow scikit-learn
```

## Dataset

The dataset used in this project is `crop_yield.csv`, containing features such as:
- Crop Type
- Crop Year
- Season
- State
- Area under cultivation
- Production
- Annual Rainfall
- Fertilizer usage
- Pesticide usage
- Yield (target variable)

## Data Preprocessing

1. **Feature Engineering**: 
   - Log transformations for skewed features (`log_Area`, `log_Production`, `log_Fertilizer`, etc.).
   - Created interaction features (`Fertilizer_Pesticide`, `Fertilizer_Rainfall`).
   
2. **Encoding**: 
   - One-hot encoding applied to categorical columns (`Crop`, `Season`, `State`).

3. **Splitting & Scaling**:
   - Dataset split into training and testing sets (80/20).
   - StandardScaler applied for feature scaling.

## Exploratory Data Analysis

Exploratory Data Analysis (EDA) involves:
- Distribution of crops, seasons, and states.
- Outlier detection in variables like `Area`, `Production`, etc., using IQR.
- Correlation matrix to analyze relationships among numeric features.
- Yearly trends in yield, area, and input usage.

## Model Architecture

The prediction model is a hybrid of Multi-Layer Perceptron (MLP) and Gated Recurrent Unit (GRU):
- **GRU**: Processes sequential data.
- **MLP**: Captures non-sequential feature relationships.
- **Combined Output**: Concatenates GRU and MLP outputs and passes through dense layers.
- **Output Layer**: Produces final yield prediction.

## Training

The model is trained using:
- Optimizer: Adam with learning rate of 0.001.
- Loss function: Mean Squared Error (MSE).
- Callbacks:
  - `ModelCheckpoint` for saving the best model.
  - `EarlyStopping` to avoid overfitting.
  - `ReduceLROnPlateau` to reduce learning rate on plateau.

## Evaluation

The model is evaluated using:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R-squared (R²)

## Results

The model achieves the following performance on the test set:
- **MAE**: 0.062
- **MSE**: 0.0097
- **R²**: 0.992

## Visualizations

1. **Training Curves**:
   - Loss and MAE values during training and validation.
   
2. **Actual vs Predicted**:
   - Plot comparing actual and predicted yields for 50 random samples.

The following graph shows a comparison of actual and predicted values for 50 random samples:

![Actual vs Predicted](actual_vs_predicted.png)

  
## Conclusion

This project provides a comprehensive approach to crop yield prediction using deep learning. The MLP-GRU model combines both sequential and non-sequential feature relationships, achieving high accuracy in predicting yield values.
