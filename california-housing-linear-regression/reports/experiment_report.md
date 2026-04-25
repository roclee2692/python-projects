# Linear Regression Experiment Report

## 1. Task

The task is to predict median house value using the California Housing dataset.

## 2. Dataset

The dataset contains housing-related features such as income, house age, average rooms, population, and geographic information.

## 3. Method

A Linear Regression model was used.
The data was split into training and testing sets with an 80:20 ratio.
StandardScaler was applied to normalize feature scales.

## 4. Evaluation Metrics

- MAE
- MSE
- RMSE
- R2

## 5. Results

The experiment produced the following test-set metrics:

| Metric | Value |
| --- | ---: |
| MAE | 0.5332 |
| MSE | 0.5559 |
| RMSE | 0.7456 |
| R2 | 0.5758 |

The learned feature coefficients sorted by absolute value were:

| Feature | Coefficient |
| --- | ---: |
| Latitude | -0.896929 |
| Longitude | -0.869842 |
| MedInc | 0.854383 |
| AveBedrms | 0.339259 |
| AveRooms | -0.294410 |
| HouseAge | 0.122546 |
| AveOccup | -0.040829 |
| Population | -0.002308 |

## 6. Analysis

Linear Regression provides a simple baseline model.
The R2 score is 0.5758, which means the model explains part of the target variance but still leaves substantial error.
This suggests that the relationship between housing features and median house value is not purely linear, or that a simple linear model cannot capture all important patterns in the data.
Latitude, longitude, and median income have large coefficients after standardization, indicating that location and income are important predictors in this baseline model.

## 7. Conclusion

This experiment completed a full regression pipeline: data loading, preprocessing, model training, evaluation, and visualization.
