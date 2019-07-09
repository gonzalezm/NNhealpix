import numpy as np
import keras as kr
import sys
import datetime
import nnhealpix
import healpix

# Directory selection
name_in = sys.argv[1]
in_dir1 = sys.argv[2]
in_dir2 = sys.argv[3]
out_dir = sys.argv[4]

date = datetie.datetime.now().strftime('%Y%m%d_%H_%M_%S')
out_dir += '{}/'.format(date) 

try:
	os.makedirs(out_dir)
except:
	pass

#Take data
Maps = np.load(in_dir1 + "/" + name_in + "_Maps.npy")
l_p = np.load(in_dir1 + "/" + name_in + "_l_p.npy")

with kr.utils.CustomObjectScope({'OrderMap':nnhealpix.layers.OrderMap}):
	model=kr.models.load_model(in_dir2 + "20190709_10_28_49_model.h5pY.File")

nside = int(np.sqrt(Maps.shape[0]/12))
print(nside)
nbmodel = l_p.shape[0]
nbtest = int(0.1*nbmodel)
ntrain = nbmodel - nbtest

x_train = Maps[:,0:ntrain]
y_train = l_p[0:ntrain]
x_test = Maps[:,ntrain:]
y_test = l_p[ntrain:]

x_train = x_train.T
xtest = x_test.T
x_train = x_train.reshape(x_train.shape[0], len(x_train[0]), 1)
x_test = x_test.reshape(x_test.shape[0], len(x_test[0]), 1)

model.summary()

checkpointer_mse = kr.callbacks.ModelCheckpoint(filepath=out_dir + date + '_weights.{epoch:02d}-{val_loss:2f}.hdf5',
	monitor='val_loss',
	verbose=0,
	save_best_only=True,
	save_weights_only=True,
	mode='min',
	period=1)

stop = kr.callbacks.EarlyStopping(monitor='val_loss',
	verbose=0,
	patience=20)

callbacks = [checkpointer_mse,stop ]

histb = model.fit(x_train, y_train,
	epoch=int(sys.argv[5]),
	batch=32,
	validation_split = 0.1,
	verbose=0,
	callbacks=callbacks,
	shuffle=True)


kr.models.save_model(model, out_dir + "/" + date + '_model.h5py.File')
np.save(out_dir + date + '_histb_loss', histb.history['loss'])
np.save(out_dir + date + '_histb_val_loss', histb.history['val_loss'])


