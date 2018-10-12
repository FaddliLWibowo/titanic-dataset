import pandas as pd
import nn 
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from keras.models import load_model


test_data = pd.read_csv("test.csv",index_col=0)

clean_test_data = test_data.drop(['Name','Ticket','Cabin'], axis=1)
final_test_data = clean_test_data.copy()
final_test_data['Age'] = final_test_data['Age'].fillna(final_test_data['Age'].median())
final_test_data['Fare'] = final_test_data['Fare'].fillna(final_test_data['Fare'].median())
final_test_data = final_test_data.dropna()

final_test_data['SibSp'] = final_test_data['SibSp'].astype('category')
final_test_data['Pclass'] = final_test_data['Pclass'].astype('category')
df = pd.get_dummies(final_test_data)

X_test = df.values


model = load_model('model')

a = model.predict_classes(X_test)
test_data['Survived'] = a
test_data.to_csv("survived.csv",header=True,index=True)