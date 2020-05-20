import tensorflow 

def CSVlogger(RUN_FOLDER):
    return tensorflow.keras.callbacks.CSVLogger(RUN_FOLDER + '/logs/epoch_history.log')

def Regular_ModelCheckpoint(RUN_FOLDER):
    return tensorflow.keras.callbacks.ModelCheckpoint(RUN_FOLDER + '/weights/model_{epoch:02d}-{val_loss:.2f}.h5', \
                                                          monitor='val_loss', verbose=0, save_weights_only=False, period=1)

def Early_Stopping(patience, min_delta=0,  verbose=1):
    return tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=min_delta, patience=patience, verbose=verbose, mode='min')

def ReduceLROnPlateau(factor, patience, min_lr, verbose=1):
    return tensorflow.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=factor, patience=patience, min_lr=min_lr, verbose=verbose, mode='min')