import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
print ("Loading dataset.....")
df=pd.read_csv('car data.csv')
print("\n data headers:")
print(df.head(2))
df=df.drop(['Car_Name'],axis=1)
current_year=2026
df['Car_Age']=current_year-df['Year']
df=df.drop('Year',axis=1)
df_processed=pd.get_dummies(df,drop_first=True)
print("\n processed data headers:")
print(df_processed.head(2))

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X=df_processed.drop('Selling_Price',axis=1)
y=df_processed['Selling_Price']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=LinearRegression()
print("\n training the model...")
model.fit(X_train,y_train)
print("\n model trained successfully")

from sklearn import metrics
print("\n evaluating the model...")
y_pred=model.predict(X_test)
error=y_test-y_pred
r2=metrics.r2_score(y_test, y_pred)
mae=metrics.mean_absolute_error(y_test, y_pred)
rmse=np.sqrt(metrics.mean_squared_error(y_test, y_pred))
print(f"R-squared Score: {r2:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue', label='Individual Cars')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', 
linestyle='--', linewidth=2, label='Perfect Prediction Line')
plt.scatter(y_pred, error, alpha=0.5, color='purple', label='Residuals')
plt.xlabel('Actual Selling Price')
plt.ylabel('Predicted Selling Price')
plt.title('Actual vs Predicted Selling Price')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.show()
