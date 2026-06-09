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