from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Features and target
x = df[['area','mainroad','basement','hotwaterheating','airconditioning','parking','prefarea','furnishingstatus']]
y = df['price']

# Convert categorical columns to numeric (One-Hot Encoding)
x = pd.get_dummies(x)

# Scale numeric data
scale = StandardScaler()
X = scale.fit_transform(x)

# Train model
reg = linear_model.LinearRegression()
reg.fit(X, y)

# Example new input (must also be in same encoded format!)
new_data = pd.DataFrame([{
    'area': 6000,
    'mainroad': 'yes',
    'basement': 'no',
    'hotwaterheating': 'no',
    'airconditioning': 'yes',
    'parking': 1,
    'prefarea': 'yes',
    'furnishingstatus': 'furnished'
}])

# Encode new data
new_data_encoded = pd.get_dummies(new_data)

# Align with training columns
new_data_encoded = new_data_encoded.reindex(columns=x.columns, fill_value=0)

# Scale
new_data_scaled = scale.transform(new_data_encoded)

# Predict
pred = reg.predict(new_data_scaled)
print(pred)

