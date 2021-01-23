from prepare_data import prep_data, batchify, prepare_train_test
from rnn_model import Model

print(1)
d = prep_data(data_file="data/fiets_1_maart_5_april_uur.csv")
print(2)
df = batchify(d, n_per_group = 24, pckle = False)

print(3)
train, test = prepare_train_test(df, 5, 0.7)

print(4)
train_x, train_y = train
test_x, test_y = test

s = train[0].shape
train_x = train_x.transpose((0, 3, 1, 2)).reshape((-1, s[3], s[1]*s[2]))
train_y = train_y.transpose((0, 2, 1))

# s = (None, 1, 24)
test_x = test_x.transpose((0, 3, 1, 2)).reshape((-1, s[3], s[1]*s[2]))
test_y = test_y.transpose((0, 2, 1))

print(5)

model = Model(training_data=(train_x, train_y), validation_data=(test_x, test_y), epochs=100, input_feature_amount=train_x.shape[2],
              output_feature_amount=train_y.shape[2])

print(6)
model.build_model()
hst = model.train()
# model.visualize_loss(hst)
model.predict(test_x[0], plot=True, y=test_y[0])
