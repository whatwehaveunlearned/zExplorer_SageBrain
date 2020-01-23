import tensorflow as tf
import tensorflow.keras # pylint: disable=import-error
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization # pylint: disable=import-error
from tensorflow.keras.regularizers import l2 # pylint: disable=import-error
from tensorflow.keras.models import Model # pylint: disable=import-error
from tensorflow.keras import backend as K # pylint: disable=import-error
from tensorflow.keras.losses import mean_squared_error # pylint: disable=import-error
from tensorflow.keras.optimizers import Adam # pylint: disable=import-error


class Encoder:

    def __init__(self, input_dim, output_dim):
        '''
        This constructor constructs and specifies the model architecture for the Encoder.
        '''

        input = Input(shape=(input_dim,)) # input to the Model
        h1 = Dense(
            150, activation='relu', use_bias=True, 
            kernel_regularizer=l2(0.001)
        )(input) # hidden layer output, has non-linearity
        encoded = Dense(output_dim, activation='linear')(h1)

        
        self.model = Model(input, encoded)
        self.model.compile(optimizer='adam', loss='mean_squared_error')


    def fit(self, input_vectors, target_vectors, epochs=60, batch_size=1):
        '''
        Fits the model parameters to the given input_vectors and target_vectors (outputs).
        The epochs parameter specifies the number of epochs to train for, and the batch_size specifies the number of examples to fit the model on per iteration. There are len(input_vectors) / batch_size iterations per epoch.
        '''

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)


        self.model.fit(
            input_vectors, target_vectors, 
            epochs=epochs, batch_size=batch_size,callbacks=[reduce_lr]
        )

    def fit_transform(self, input_vectors, target_vectors, epochs=40, batch_size=1):
        '''
        Fits the model parameters to the givent inputs and outputs, and returns the
        predicted embeddings (tranforms) from the model.
        '''

        self.fit(input_vectors, target_vectors, epochs, batch_size)
        embeddings = self.model.predict(input_vectors)
        return embeddings

    def transform(self, input_vectors):
        '''
        Transforms the given input_vectors into the low-Dimensional embedding
        space by calling the models predict method on the input_vectors, and
        returning the results
        '''

        embeddings = self.model.predict(input_vectors)
        return embeddings
