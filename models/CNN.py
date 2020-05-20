from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

class Basic_CNN():
    def __init__(self
        , input_dim
        , conv_filters
        , conv_kernel_size
        , fc_layer_size
        ):
        
        self.name = 'Basic_CNN'
        self.input_dim = input_dim
        self.conv_filters = conv_filters
        self.conv_kernel_size = conv_kernel_size
        self.fc_layer_size = fc_layer_size

        self.n_layers_conv = len(conv_filters)
        self.n_layers_dense = len(fc_layer_size)

        self._build() 
    
    def _build(self):
        
        inp = Input(shape=self.input_dim, name='input')
        
        x = inp
        for i in range(self.n_layers_conv):
            
            conv_layer = Conv2D(
                filters = self.conv_filters[i]
                , kernel_size = self.conv_kernel_size[i]
                , padding = 'same'
                , activation='relu'
                , name = 'conv_' + str(i)
                )
            
            x = conv_layer(x)
            x = MaxPooling2D((2, 2))(x)
            
        x = Flatten()(x)
        for i in range(self.n_layers_dense):
            
            x = Dense(self.fc_layer_size[i], activation='relu', name = 'dense_' + str(i))(x)
        
        out = Dense(10, activation='softmax')(x)

        model_input = inp
        model_output = out
        
        self.model = Model(model_input, model_output)   
    
    def compile(self, learning_rate):
        self.learning_rate = learning_rate

        optimizer = Adam(lr=learning_rate)

        self.model.compile(optimizer=optimizer, 
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])     
    
    def save_model(self, folder, checkpoint_name):
        
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.model.save( os.path.join(folder, checkpoint_name, '.h5') )
        
    def load_model(self, folder, checkpoint_name):
        
        self.model = tensorflow.keras.models.load_model( os.path.join(folder, checkpoint_name, '.h5') )
        
    
    
    def train(self, train_generator, validation_generator, epochs=1, shuffle=True, verbose=1, callbacks=[]):

        history = self.model.fit_generator(generator=train_generator,
                              validation_data=validation_generator,
                              epochs=epochs,
                              callbacks=callbacks,
                              max_queue_size = 1000,
                              workers = 32,
                              use_multiprocessing = False,
                              verbose=verbose)