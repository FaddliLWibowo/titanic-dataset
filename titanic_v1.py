import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("train.csv", index_col=0)
test_data = pd.read_csv("test.csv",index_col=0)
print("Cleaning Data...")

clean_data = data.drop(['Name','Ticket','Cabin'], axis=1)
final_data = clean_data.copy()
final_data['Age'] = final_data['Age'].fillna(final_data['Age'].median())
final_data = final_data.dropna()

clean_test_data = test_data.drop(['Name','Ticket','Cabin'], axis=1)
final_test_data = clean_test_data.copy()
final_test_data['Age'] = final_test_data['Age'].fillna(final_test_data['Age'].median())
final_test_data['Fare'] = final_test_data['Fare'].fillna(final_test_data['Fare'].median())
final_test_data = final_test_data.dropna()

male = 0
female = 1
labels = ['male','female']
encoder = preprocessing.LabelEncoder()
encoder.fit(labels)
final_data['Sex'] = encoder.fit_transform(final_data['Sex'])
final_test_data['Sex'] = encoder.fit_transform(final_test_data['Sex'])

S = 0
C = 1
Q = 2
label = ['S','C','Q']
Embarkedencoder = preprocessing.LabelEncoder()
Embarkedencoder.fit(label)
final_data['Embarked'] = Embarkedencoder.fit_transform(final_data['Embarked'].astype(str))
final_test_data['Embarked'] = Embarkedencoder.fit_transform(final_test_data['Embarked'].astype(str))

test_label = final_data.loc[:,'Survived']
X = final_data.loc[:, final_data.columns != 'Survived']
y = final_data.loc[:, 'Survived']
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X,y)

titanic_prediction = final_test_data.copy()
knn.predict(titanic_prediction)
print("Predicting some stuff...")
titanic_prediction['Survived'] = knn.predict(titanic_prediction)
print("Here is the output")
titanic_prediction.to_csv('hello.csv', header=True)
