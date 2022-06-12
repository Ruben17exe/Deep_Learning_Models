import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer
from keras.models import Sequential
from keras.layers import InputLayer, Dense
from keras.optimizers import Adam


def design_model(features):
    model = Sequential(name="my_first_model")
    num_features = features.shape[1]
    # Weight: Decides how much influence the input will have on the output
    # Biases: Always have the value of 1. Guarantees that even when all the inputs are zeros there will still be an
    #         activation in the neuron
    input_layers = InputLayer(input_shape=(num_features,))
    model.add(input_layers)
    # This command adds a hidden layer to a model instance
    model.add(Dense(128, activation="relu"))  # 128 hidden units, a "relu" activation function
    # This command adds an output layer with one neuron to a model instance
    model.add(Dense(1))
    # If it is set too high, the optimizer will make large jumps and possibly miss solutions. If it is set too low
    # the process will be too slow and might not converge to a desirable solution with the allotted time. Usually = 0.01
    opt = Adam(learning_rate=0.01)
    # loss: Denotes the measure of learning success and the lower the loss the better the performance. In the case of
    # regression, the most often used loss function is the Mean Squared Error.
    # Additionally, we want to observe the progress of the Mean Absolute Error while training the model.
    model.compile(loss="mse", metrics=["mae"], optimizer=opt)
    return model


df = pd.read_csv("Insurance.csv")
features = df.iloc[:, 0:6]
labels = df.iloc[:, -1]
features = pd.get_dummies(features)  # Convert the text data to numerical
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=42)

# Normalize the data
ct = ColumnTransformer([('normalize', Normalizer(), ['age', 'bmi', 'children'])], remainder='passthrough')
x_train_norm = pd.DataFrame(ct.fit_transform(x_train), columns=features.columns)
x_test_norm = pd.DataFrame(ct.transform(x_test), columns=features.columns)

model = design_model(x_train_norm)
print(model.summary(()))
model.fit(x_train_norm, y_train, epochs=40, batch_size=1, verbose=1)
val_mse, val_mae = model.evaluate(x_test_norm, y_test, verbose=0)
print("MAE: ", val_mae)
