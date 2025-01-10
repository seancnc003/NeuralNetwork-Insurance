# data processing and visualization
import pandas as pnds
import numpy as nmpy
import matplotlib.pyplot as pyplot

# scikit-learn libraries
from sklearn.model_selection import RandomizedSearchCV, KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error


# read the csv file
data = pnds.read_csv('insurance.csv')

# remove the 'region' feature similar to the previous model
data = data.drop('region', axis=1)

# separate the dataset into features and target
X = data.drop('charges', axis=1)
y = data['charges']

# identify categorical and numerical columns
categorical_cols = ['sex', 'smoker'] 
numerical_cols = ['age', 'bmi', 'children']

# define the preprocessor: standardize numerical data and
# one-hot encode categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False),
         categorical_cols)]
)

# scale the target variable
y = y.values.reshape(-1, 1)
y_scaler = StandardScaler()
y_scaled = y_scaler.fit_transform(y).ravel()

# define the pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('mlp', MLPRegressor(random_state=42))
])

# define a focused hyperparameter grid for randomizedsearchcv
param_distributions = {
    'mlp__hidden_layer_sizes': [(50,), (100,), (100, 50)],
    'mlp__activation': ['relu'],  # focus on 'relu' initially
    'mlp__solver': ['adam'],      # focus on 'adam' for efficiency
    'mlp__alpha': [0.0001, 0.001],
    'mlp__learning_rate_init': [0.001, 0.01],
    'mlp__batch_size': [32, 64],
    'mlp__max_iter': [500],       # keep max_iter consistent
    'mlp__early_stopping': [True],  # enable early stopping
    'mlp__tol': [1e-4]            # tolerance for convergence
}

# total combinations: 3 (hidden_layer_sizes) × 1 (activation) × 1 (solver) ×
# 2 (alpha) × 2 (learning_rate_init) × 2 (batch_size) × 1 (max_iter) ×
# 1 (early_stopping) × 1 (tol) = 24

# define 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# initialize randomizedsearchcv
random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_distributions,
    n_iter=24,              # cover all possible combinations
    cv=kf,
    scoring='neg_mean_squared_error',
    n_jobs=-1,              # utilize all available cpu cores
    verbose=2,              # verbosity level to monitor progress
    random_state=42         # ensures reproducibility
)

# fit randomizedsearchcv
print("Starting hyperparameter search with 10-fold cross-validation...")
random_search.fit(X, y_scaled)
print("Hyperparameter search completed.")

# display the best parameters
print("\nBest Hyperparameters:")
print(random_search.best_params_)

# extract the best model
best_model = random_search.best_estimator_

# initialize lists to store per-fold results
results_list = []

# initialize kfold for manual cross-validation to extract per-fold metrics
for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
    print(f"\nProcessing Fold {fold}...")
    
    # split data
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train_scaled, y_test_scaled = y_scaled[train_index], y_scaled[test_index]
    y_train_orig, y_test_orig = y[train_index].ravel(), y[test_index].ravel()
    
    # fit the model on the training data
    best_model.fit(X_train, y_train_scaled)
    
    # predict on the test data
    y_pred_scaled = best_model.predict(X_test)
    
    # inverse transform predictions and actuals to original scale
    y_pred_orig = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    
    # calculate rmse
    rmse_standardized = nmpy.sqrt(mean_squared_error(y_test_scaled, y_pred_scaled))
    rmse_dollars = nmpy.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
    
    # extract feature weights from the first hidden layer
    # we'll take the average of the absolute weights for each input feature
    # across all neurons in the first hidden layer
    # note: this is a rudimentary method and doesn't capture the complexity
    # of neural network feature interactions
    first_hidden_weights = best_model.named_steps['mlp'].coefs_[0]  # shape: (n_features, n_hidden_neurons)
    feature_importance = nmpy.mean(nmpy.abs(first_hidden_weights), axis=1)
    
    # create a dictionary for the current fold
    fold_dict = {
        'Fold': fold,
        'RMSE (Standardized)': rmse_standardized,
        'RMSE (Dollars)': rmse_dollars
    }
    
    # map feature importances to feature names
    feature_names_num = numerical_cols
    feature_names_cat = best_model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_cols)
    feature_names = nmpy.concatenate([feature_names_num, feature_names_cat])
    
    for feature, importance in zip(feature_names, feature_importance):
        fold_dict[feature] = importance
    
    # append the fold results to the list
    results_list.append(fold_dict)

# create a dataframe from the results
results_df = pnds.DataFrame(results_list)

# rearrange columns to match desired report structure
# assuming the categorical features after one-hot encoding are 'sex_male' and 'smoker_yes'
desired_columns = [
    'Fold',
    'age',
    'bmi',
    'children',
    'sex_male',
    'smoker_yes',
    'RMSE (Standardized)',
    'RMSE (Dollars)'
]

# ensure all desired columns are present
for col in desired_columns:
    if col not in results_df.columns:
        results_df[col] = nmpy.nan  # fill missing columns with nan

# reorder the columns
results_df = results_df[desired_columns]

# rename columns for better readability
results_df.rename(columns={
    'age': 'Feature 1 (age)',
    'bmi': 'Feature 2 (bmi)',
    'children': 'Feature 3 (children)',
    'sex_male': 'Feature 4 (sex_male)',
    'smoker_yes': 'Feature 5 (smoker_yes)',
    'RMSE (Standardized)': 'RMSE (Standardized)',
    'RMSE (Dollars)': 'RMSE (Dollars)'
}, inplace=True)

# display the report
print("\nDetailed Per-Fold Report:")
print(results_df.to_string(index=False))

# perform cross-validation predictions for plotting
print("\nGenerating cross-validation predictions for plotting...")
y_preds_scaled = cross_val_predict(best_model, X, y_scaled, cv=kf, n_jobs=-1)
y_tests_scaled = y_scaled  # all test sets are covered

# inverse transform the predictions and actual values to original scale
y_preds_orig = y_scaler.inverse_transform(y_preds_scaled.reshape(-1, 1)).ravel()
y_tests_orig = y_scaler.inverse_transform(y_tests_scaled.reshape(-1, 1)).ravel()

# calculate overall rmse
overall_rmse_standardized = nmpy.sqrt(mean_squared_error(y_tests_scaled, y_preds_scaled))
overall_rmse_dollars = nmpy.sqrt(mean_squared_error(y_tests_orig, y_preds_orig))
print(f"\nOverall RMSE (Standardized): {overall_rmse_standardized:.4f}")
print(f"Overall RMSE (Dollars): ${overall_rmse_dollars:.2f}")

# plot predicted vs. actual values (dollars)
pyplot.figure(figsize=(10, 6))
pyplot.scatter(range(len(y_tests_orig)), y_tests_orig, color='red', alpha=0.6, label='Actual Charges')
pyplot.scatter(range(len(y_preds_orig)), y_preds_orig, color='blue', alpha=0.6, label='Predicted Charges')
pyplot.xlabel('Sample Index')
pyplot.ylabel('Charges ($)')
pyplot.title('Predicted vs. Actual Charges (in Dollars)')
pyplot.legend()
pyplot.show()
