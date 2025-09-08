d = {'UK': 0, 'USA': 1, 'N': 2}
df['Nationality'] = df['Nationality'].map(d)
d = {'YES': 1, 'NO': 0}
df['Go'] = df['Go'].map(d)

print(df)

X = df[['Age', 'Experience', 'Rank', 'Nationality']]
y = df['Go']

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# Create and fit our decision tree classifier
model = DecisionTreeClassifier()
model.fit(X, y)

tree.plot_tree(model, feature_names=['Age', 'Experience', 'Rank', 'Nationality'])
plt.show()

print(model.predict([[40, 10, 7, 1]]))

###########################################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error

np.random.seed(2)

# Generate synthetic data
x = np.random.normal(3, 1, 100)
y = np.random.normal(150, 40, 100) / x
X = x.reshape(-1, 1)

# Train-test split
train_X, test_X = X[:80], X[80:]
train_y, test_y = y[:80], y[80:]

# Fit DecisionTreeRegressor
reg_tree = DecisionTreeRegressor(random_state=42, max_depth=None)
reg_tree.fit(train_X, train_y)

# Predict and evaluate
train_pred = reg_tree.predict(train_X)
test_pred = reg_tree.predict(test_X)

print("Train R2:", r2_score(train_y, train_pred))
print("Test R2 :", r2_score(test_y, test_pred))
print("Train MSE:", mean_squared_error(train_y, train_pred))
print("Test MSE :", mean_squared_error(test_y, test_pred))

# Cross-validated R2
cv_scores = cross_val_score(reg_tree, X, y, cv=5, scoring="r2")
print("Cross-validated R2 scores:", cv_scores)
print("CV mean R2:", cv_scores.mean())

# Optional: visualize the tree (regression tree)
plt.figure(figsize=(8,6))
plot_tree(reg_tree, feature_names=["x"], filled=True, rounded=True)
plt.show()


###########################grid search

# Tune hyperparameters with GridSearchCV
param_grid = {
    'max_depth': [1, 2, 3, 4, 5, 6, None],
    'min_samples_leaf': [1, 2, 3, 5, 10]
}

grid = GridSearchCV(DecisionTreeRegressor(random_state=42), param_grid,
                    cv=5, scoring='r2', n_jobs=-1, verbose=0)
grid.fit(train_X, train_y)

print("Best params (GridSearch):", grid.best_params_)
print("Best CV score (GridSearch):", grid.best_score_)
print()

# Use best estimator found
dt_best = grid.best_estimator_
dt_best_train_pred = dt_best.predict(train_X)
dt_best_test_pred = dt_best.predict(test_X)
print_metrics("Decision Tree (best from GridSearch)", train_y, dt_best_train_pred, test_y, dt_best_test_pred)

