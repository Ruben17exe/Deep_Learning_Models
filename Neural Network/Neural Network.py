import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
import numpy as np

data = pd.read_csv("heart_failure.csv")
# print(data.info())
# print(Counter(data["death_event"]))
x = data.iloc[:, 0:13]
y = data["death_event"]
x = pd.get_dummies(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=101)
ct = ColumnTransformer([("StandardScaler", StandardScaler(), ['age','creatinine_phosphokinase','ejection_fraction','platelets','serum_creatinine','serum_sodium','time'])], remainder="passthrough")
x_train = ct.fit_transform(x_train)
x_test = ct.transform(x_test)

le = LabelEncoder()
y_train = le.fit_transform(y_train.astype(str))
y_test = le.transform(y_test.astype(str))
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()
model.add(InputLayer(input_shape=(x_train.shape[1],)))
model.add(Dense(12, activation="relu"))
model.add(Dense(2, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, epochs=100, batch_size=16)
loss, acc = model.evaluate(x_test, y_test)
y_estimate = model.predict(x_test)
y_estimate = np.argmax(y_estimate)
y_true = np.argmax(y_test)
