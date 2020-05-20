from tensorflow.keras.preprocessing.image import ImageDataGenerator

def data_generators(train_dir, test_dir, batch_size=32, augment=True):
    
    if augment == True:
        train_datagen = ImageDataGenerator(rescale=1./255,
                                  fill_mode = "nearest",
                                  zoom_range = 0.2,
                                  width_shift_range = 0.2,
                                  height_shift_range=0.2,
                                  rotation_range=30)
    else:
        train_datagen = ImageDataGenerator(rescale=1./255)

    test_datagen = ImageDataGenerator(rescale=1./255)


    train_generator = train_datagen.flow_from_directory(
        directory=train_dir,
        target_size=(32, 32),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        directory=test_dir,
        target_size=(32, 32),
        batch_size=batch_size,
        class_mode='categorical')
    
    return (train_generator, validation_generator)