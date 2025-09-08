import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# 2) Prepare target y (map 'M'->1, 'B'->0)
y = df['diagnosis'].map({"M": 1, "B": 0})

# 3) Choose single feature x = radius_mean
x = df['radius_mean'].astype(float)

# 4) Simple preprocessing: fill missing values (if any)
# (for a single column it's simplest to fill with the column mean)
x = x.fillna(x.mean())

# 5) Scale the feature (important for some models and to interpret coeffs)
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x.values.reshape(-1, 1))  # shape (n_samples, 1)


X_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42, stratify=y)

# 7) Train logistic regression
model = LogisticRegression(solver='liblinear', random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(x_test)
y_prob = model.predict_proba(x_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))
print("\nClassification report:\n", classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# 9) Predict a new sample (e.g., radius_mean = 3.46)
new_value = 3.46
new_scaled = scaler.transform(np.array([[new_value]]))   # must scale exactly the same way
pred_label = model.predict(new_scaled)[0]
pred_prob = model.predict_proba(new_scaled)[0, 1]
print(f"\nFor radius_mean = {new_value}: predicted class = {pred_label} (1=malignant), prob_malignant = {pred_prob:.4f}")

# 10) Optional: quick check whether radius_mean is a reasonable single feature
# Map target to numeric first (already mapped above); compute correlation
y_num = y.astype(int)
corr = df['radius_mean'].astype(float).corr(y_num)
print(f"\nCorrelation between radius_mean and diagnosis (0/1): {corr:.3f}")
if abs(corr) < 0.2:
    print("Note: correlation is low — you might want to include more features.")
else:
    print("radius_mean has notable correlation with diagnosis; single-feature model can be informative.")
