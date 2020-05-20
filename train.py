#!/usr/bin/env python
import os

from utils.loader import data_generators
from utils.callbacks import CSVlogger, Regular_ModelCheckpoint, Early_Stopping, ReduceLROnPlateau
from models.CNN import Basic_CNN


### Set Params 
import argparse
parser = argparse.ArgumentParser(description="sample script")
parser.add_argument('--checkpoint_dir', help='root directory', default='/home/khan74/checkpoint/')
parser.add_argument('--run_id', type=str, help='run ID', default='0001')
parser.add_argument('--data_dir', help='root directory', default='/home/khan74/data/')
parser.add_argument('--learning_rate', type=float, help='learning rate', default=0.001)
parser.add_argument('--batch_size', type=int, help='batch size', default=64)
parser.add_argument('--data_aug', type=bool, help='data augmentation', default=True)
parser.add_argument('--epochs', type=int, help='num of epochs to train', default=1)
parser.add_argument('--shuffle', type=bool, help='whether to shuffle data', default=True)
parser.add_argument('--verbose', type=int, help='verbosity', default=1)
parser.add_argument('--mode', type=str, help='mode: build or load', default='build')


args = parser.parse_args()

checkpoint_dir = args.checkpoint_dir
RUN_ID = args.run_id
DATA_DIR = args.data_dir
LEARNING_RATE = args.learning_rate
BATCH_SIZE = args.batch_size
DATA_AUGMENTATION = args.data_aug
EPOCHS = args.epochs
SHUFFLE = args.shuffle
VERBOSE = args.verbose
MODE = args.mode

RUN_FOLDER = os.path.join(checkpoint_dir , 'runs/{}/'.format(RUN_ID))

if not os.path.exists(RUN_FOLDER):
    os.makedirs(RUN_FOLDER)
    os.makedirs(RUN_FOLDER + '/weights/')
    os.makedirs(RUN_FOLDER +'/logs/')
    
    
### Load data 
train_generator, validation_generator = data_generators(train_dir=DATA_DIR+'/train/', 
                                                             test_dir=DATA_DIR+'/test/', 
                                                             batch_size=BATCH_SIZE,
                                                             augment=DATA_AUGMENTATION)

### Define Model 

if MODE ==  'build': #'load' #
    CNN = Basic_CNN(
        input_dim=(32, 32, 3)
        , conv_filters = [32,64,64]
        , conv_kernel_size = [3,3,3]
        , fc_layer_size = [64]
        )
else:
    pass


### Train the Model

CNN.compile(LEARNING_RATE)

callbacks = [
    CSVlogger(RUN_FOLDER),
    Regular_ModelCheckpoint(RUN_FOLDER),
    Early_Stopping(patience=7, verbose=1),
    ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-10, verbose=1),  
] 

CNN.train(train_generator, validation_generator, epochs=EPOCHS, shuffle=SHUFFLE, verbose=VERBOSE, callbacks=callbacks)