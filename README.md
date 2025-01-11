# Neural Network Insurance Prediction

A project leveraging neural networks to predict insurance charges based on personal attributes. This project demonstrates the implementation of a regression task using Scikit-Learn's MLPRegressor and evaluates its performance compared to linear regression.

---

## Features
- **Data Preprocessing**:
  - Standardizes numerical features like age, BMI, and children using `StandardScaler`.
  - Encodes categorical variables (sex and smoker status) using `OneHotEncoder`.
- **Neural Network Architecture**:
  - Input Layer: 5 neurons for the selected features.
  - Hidden Layers:
    - 1st Layer: 100 neurons with ReLU activation.
    - 2nd Layer: 50 neurons with ReLU activation.
  - Output Layer: 1 neuron for predicting insurance charges.
- **Model Optimization**:
  - Hyperparameter tuning with `RandomizedSearchCV`.
  - Early stopping to avoid overfitting.
- **Evaluation Metrics**:
  - Root Mean Squared Error (RMSE) in standardized units and dollar amounts.
  - Performance comparison with a linear regression model.

---

## Dataset
- **Source**: Insurance dataset.
- **Features Used**:
  - Age, BMI, number of children, sex, and smoker status.
  - Region was excluded as it showed little impact on the prediction.

---

## Results
- **Neural Network Performance**:
  - Average RMSE (Standardized): 0.3881
  - Average RMSE (Dollars): $4698.66
- **Comparison with Linear Regression**:
  - Neural Network RMSE is ~22.6% lower than Linear Regression.
  - Significant improvement in handling outliers.
- **Tradeoffs**:
  - Neural Network offers better performance but is computationally more expensive compared to Linear Regression.

---

## Model Architecture
| Layer         | Neurons | Activation Function |
|---------------|---------|----------------------|
| Input Layer   | 5       | -                    |
| Hidden Layer 1| 100     | ReLU                |
| Hidden Layer 2| 50      | ReLU                |
| Output Layer  | 1       | Linear              |

- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam with a learning rate of 0.001
- **Regularization**: L2 Penalty (0.001)

---

## Libraries Used
- **Core Libraries**:
  - `pandas`, `numpy`, `matplotlib`
- **Scikit-Learn Utilities**:
  - `StandardScaler`, `OneHotEncoder`, `MLPRegressor`
  - `RandomizedSearchCV`, `cross_val_predict`
  - `KFold` for 10-fold cross-validation

---

## Resume Bullet Points
Neural Network Insurance Charges Model (Python/Scikit) | Machine Learning    

* Implemented a predictive neural network model for linear regression to estimate insurance charges 
* Analyzed over 9,000 data points with a standardized RMSE of 0.3881 as the average difference from actual values.


## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/NeuralNetwork-Insurance.git
