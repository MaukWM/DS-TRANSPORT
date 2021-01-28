import pickle

from evaluate_model import generate_naive_prediction

with open("testtrain5nodes.p", "rb") as f:
    train, test, mean_y = pickle.load(f)

# Set test and train
train_x, train_y = train
test_x, test_y = test

generate_naive_prediction(test_y, mean_y)
