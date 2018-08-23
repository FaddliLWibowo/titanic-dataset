import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv("train.csv", index_col=0)
test_data = pd.read_csv("test.csv",index_col=0)

print("Cleaning Data...")

clean_data = data.drop(['Name','Ticket'], axis=1)
final_data = clean_data.copy()
final_data['Age'] = final_data['Age'].fillna(final_data['Age'].median())
final_data['Deck'] = final_data['Cabin'].str.slice(0,1)
final_data['Room'] = final_data['Cabin'].str.slice(1,5).str.extract('([0-9]+)',expand=False).astype('float')
final_data = final_data.drop(['Cabin'], axis = 1)
final_data ['Deck'] = final_data['Deck'].fillna('U')
final_data['Room'] = final_data['Room'].fillna(final_data['Room'].median())

clean_test_data = test_data.drop(['Name','Ticket'], axis=1)
final_test_data = clean_test_data.copy()
final_test_data['Age'] = final_test_data['Age'].fillna(final_test_data['Age'].median())
final_test_data['Deck'] = final_test_data['Cabin'].str.slice(0,1)
final_test_data['Room'] = final_test_data['Cabin'].str.slice(1,5).str.extract('([0-9]+)',expand=False).astype('float')
final_test_data = final_test_data.drop(['Cabin'], axis = 1)
final_test_data ['Deck'] = final_test_data['Deck'].fillna('U')
final_test_data['Fare'] = final_test_data['Fare'].fillna(7.225)
final_test_data['Room'] = final_test_data['Room'].fillna(final_test_data['Room'].median())

A = 0
B = 1
C = 2
D = 3
E = 4
F = 5
G = 6
T = 7
U = 8

labels_cabin = ['U','C','B','D','E','A','F','G','T']
Cabinencoder = preprocessing.LabelEncoder()
Cabinencoder.fit(labels_cabin)
final_data['Deck'] = Cabinencoder.fit_transform(final_data['Deck'])

A = 0
B = 1
C = 2
D = 3
E = 4
F = 5
G = 6
U = 7

labels_cabin = ['U','C','B','D','E','A','F','G']
Cabinencoder = preprocessing.LabelEncoder()
Cabinencoder.fit(labels_cabin)
final_test_data['Deck'] = Cabinencoder.fit_transform(final_test_data['Deck'])

female = 0
male = 1
labels = ['male','female']
encoder = preprocessing.LabelEncoder()
encoder.fit(labels)
final_data['Sex'] = encoder.fit_transform(final_data['Sex'])
final_test_data['Sex'] = encoder.fit_transform(final_test_data['Sex'])

C = 0
Q = 1
S = 2
label = ['S','C','Q']
Embarkedencoder = preprocessing.LabelEncoder()
Embarkedencoder.fit(label)
final_data['Embarked'] = Embarkedencoder.fit_transform(final_data['Embarked'].astype(str))
final_test_data['Embarked'] = Embarkedencoder.fit_transform(final_test_data['Embarked'].astype(str))

X = final_data.loc[:, final_data.columns != 'Survived']
y = final_data.loc[:, 'Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.75, random_state=0)

test_knn = KNeighborsClassifier(n_neighbors=10)
clf = test_knn.fit(X_train,y_train)
accuracy = round(clf.score(X_test,y_test),2) * 100

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X,y)

titanic_prediction = final_test_data.copy()
print("Predicting some stuff...")
titanic_prediction['Survived'] = knn.predict(titanic_prediction)
print("Here is the output")
titanic_prediction = titanic_prediction.drop(['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Deck','Room'], axis=1)
titanic_prediction.to_csv('hello2.csv', header=True)
print('The accuracy of this model is ' + '{}'.format(accuracy) +'%')