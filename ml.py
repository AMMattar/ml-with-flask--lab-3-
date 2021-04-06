import pandas as pd

df = pd.read_csv('diabetes.csv')

x = df.iloc [: ,:8].values
y = df.iloc [: ,-1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 17, metric = 'minkowski')
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

from sklearn.pipeline import Pipeline
pipeline = Pipeline(steps=[('sc', sc),('classifier', classifier)])

#from joblib import dump
#dump(pipeline, filename="my_ML.joblib")
#if error occured use this liberary
import pickle
pickle.dump(pipeline, open('ml.pkl','wb'))
