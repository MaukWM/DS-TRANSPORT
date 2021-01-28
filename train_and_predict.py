from prepare_data import prep_data, batchify, prepare_train_test
from rnn_model import Model
import pandas as pd
print(1)
d = prep_data(data_file="data/fiets_1_maart_5_april_uur.csv")
print(2)
df = batchify(d, n_per_group = 24, pckle = False)
df.to_pickle("data/windowed_data.pkl")

# df = pd.read_pickle('data/windowed_data.pkl')
print(3)
train, test = prepare_train_test(df, 8, 0.2)

print(4)

train_x, train_y = train
test_x, test_y = test


s = train[0].shape
train_x = train_x.transpose((0, 3, 1, 2)).reshape((-1, s[3], s[1]*s[2]))
train_y = train_y.transpose((0, 2, 1))

# s = (None, 1, 24)
test_x = test_x.transpose((0, 3, 1, 2)).reshape((-1, s[3], s[1]*s[2]))
test_y = test_y.transpose((0, 2, 1))

print(train_x.shape, train_y.shape)
print(test_x.shape, test_y.shape)
print(5)

model = Model(training_data=(train_x, train_y), validation_data=(test_x, test_y), epochs=100, input_feature_amount=train_x.shape[2],
              output_feature_amount=train_y.shape[2])

print(6)
model.build_model()
# model.predict(train_x[0], plot=True, y=train_y[0])
hst = model.train()
# model.visualize_loss(hst)
# model.model.load_model("data/checkpoint2021-01-23_211350.h5")
model.predict(test_x[40], plot=True, y=test_y[40])
model.predict(test_x[41], plot=True, y=test_y[41])
model.predict(test_x[42], plot=True, y=test_y[42])
model.predict(test_x[43], plot=True, y=test_y[43])
model.predict(test_x[44], plot=True, y=test_y[44])

# Evaluate the model
loss, acc = model.model.evaluate(test_x, test_y, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))
