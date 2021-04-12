import pandas
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

data = pandas.read_csv('dataAllNoRotate.csv')
array = data.values
X = array[:,0:42]
Y = array[:,42]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.30,stratify=Y ,random_state=6)
# Fit the model on training set
model = MLPClassifier(max_iter=120000).fit(X_train, Y_train)
# model.predict_proba(X_test[:1])
# model.predict(X_test[:5, :])

# save the model to disk
filename = 'loaded_dataAllNoRotate.sav'
pickle.dump(model, open(filename, 'wb'))

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)
print(result)
