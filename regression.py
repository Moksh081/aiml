#polynomial_reg:

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

np.random.seed(2)

# Generate synthetic data
x = np.random.normal(3, 1, 100)
y = np.random.normal(150, 40, 100) / x

# Split into train & test
train_x = x[:80]
train_y = y[:80]
test_x = x[80:]
test_y = y[80:]

# Fit polynomial regression model (degree=4)
mymodel = np.poly1d(np.polyfit(train_x, train_y, 4))

# Line for plotting
myline = np.linspace(0, 6, 100)

# Plot training data + fitted curve
plt.scatter(train_x, train_y, label="Train Data")
plt.plot(myline, mymodel(myline), color="red", label="Model")
plt.legend()
plt.show()

# Evaluate regression model
pred_y = mymodel(test_x)

print("R2 Score:", r2_score(test_y, pred_y))
print("MSE:", mean_squared_error(test_y, pred_y))
print("MAE:", mean_absolute_error(test_y, pred_y))

r2 = r2_score(test_y, mymodel(test_x))

print(r2)

print(mymodel(5))


############multiple 
import pandas as pd
df = pd.read_csv('car.csv')
print(df)
x = df[['Weight', 'Volume']]
y = df['CO']

from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
X = scale.fit_transform(x)
reg = linear_model.LinearRegression()
reg.fit(X,y)
scaled = scale.fit_transform([[2300,1.3]])
predict_co = reg.predict([scaled[0]])
predict_co

#print(reg.coef_)

