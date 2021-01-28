from prepare_data import prep_data, batchify, prepare_train_test
from rnn_model import Model
import pickle
import pandas as pd

generate_data = True
if generate_data:
    print(1)
    d = prep_data(data_file="data/fiets_1_maart_5_april_uur.csv")
    print(2)
    df = batchify(d, n_per_group = 24, pckle = False)
    df.to_pickle("data/windowed_data.pkl")

    # df = pd.read_pickle('data/windowed_data.pkl')
    print(3)

    to_pkl = prepare_train_test(df, 8, 0.7)
    train_x, train_y, test_x, test_y, means_y = to_pkl

    pickle.dump(to_pkl, open("testtrain.p", "wb"))
    with open("testtrain.p", "wb") as f:
        pickle.dump(to_pkl, f)
else:
    with open("testtrain.p", "rb") as f:
        train_x, train_y, test_x, test_y, means_y = pickle.load(f)

print(4)


print(train_x.shape, train_y.shape)
print(test_x.shape, test_y.shape)
print(means_y.shape)
exit()
print(5)

model = Model(training_data=(train_x, train_y), validation_data=(test_x, test_y), epochs=100, input_feature_amount=train_x.shape[2],
              output_feature_amount=train_y.shape[2])

print(6)
model.build_model()
model.predict(test_x[0], plot=True, y=test_y[0])
hst = model.train()
# model.visualize_loss(hst)
model.predict(test_y[0], plot=True, y=test_y[0])

# Evaluate the model
loss, acc = model.model.evaluate(test_x, test_y, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))
