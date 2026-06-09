import pandas as pd
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