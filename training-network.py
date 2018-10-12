import pandas as pd
import nn 
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

data = pd.read_csv("train.csv", index_col=0)
test_data = pd.read_csv("test.csv",index_col=0)

print("Cleaning Data...")

clean_data = data.drop(['Name','Ticket','Cabin'], axis=1)
final_data = clean_data.copy()
final_data['Age'] = final_data['Age'].fillna(final_data['Age'].median())
final_data = final_data.dropna()
final_data['SibSp'] = final_data['SibSp'].astype('category')
final_data['Pclass'] = final_data['Pclass'].astype('category')
df = pd.get_dummies(final_data)

X = df.iloc[:, df.columns != 'Survived'].values
y = df['Survived']


clean_test_data = test_data.drop(['Name','Ticket','Cabin'], axis=1)
final_test_data = clean_test_data.copy()
final_test_data['Age'] = final_test_data['Age'].fillna(final_test_data['Age'].median())
final_test_data['Fare'] = final_test_data['Fare'].fillna(final_test_data['Fare'].median())
final_test_data = final_test_data.dropna()

final_test_data['SibSp'] = final_test_data['SibSp'].astype('category')
final_test_data['Parch'] = final_test_data['Parch'].astype('category')
final_test_data['Pclass'] = final_test_data['Pclass'].astype('category')

X_test = final_test_data.values


input_dim = X.shape[1]

model = nn.build(input_dim)

EPOCHS = 80
LR = 0.001
BS = len(X) // 100

model = nn.build(input_dim)

model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

history=model.fit(x=X,y=y,verbose=1,
                  epochs=EPOCHS,batch_size=BS,
                  validation_split=0.2,shuffle=True)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()890


model.save("alt_model")