
import tensorflow 

class BaseModel(object):
    def __init__(self, config):
        self.config = config
        self.model = None

class Model_gep(BaseModel):
    """
    Constructor for the Generator model
    Parameters
    ----------
    Returns
    -------
    This function initializes the architecture for the Generator
    """
    def __init__(self, config):
        super(Model_gep, self).__init__(config)
        self.build_model()

    def build_model(self):

        input_data = tensorflow.keras.layers.Input(shape=(self.config.inp_dimension,))

        encoder = tensorflow.keras.layers.Dense(2048,activation='relu')(input_data)
        encoder = tensorflow.keras.layers.BatchNormalization()(encoder)
        # encoder = tensorflow.keras.layers.Dropout(0)(encoder) 
        encoder = tensorflow.keras.layers.Dense(self.config.dense_1,activation='relu')(encoder)
        encoder = tensorflow.keras.layers.BatchNormalization()(encoder)
        # encoder = tensorflow.keras.layers.Dropout(0)(encoder) 
        encoder = tensorflow.keras.layers.Dense(self.config.dense_2,activation='relu')(encoder)
        encoder = tensorflow.keras.layers.BatchNormalization()(encoder)
        # encoder = tensorflow.keras.layers.Dropout(0)(encoder) 
        encoder = tensorflow.keras.layers.Dense(self.config.dense_3,activation='relu')(encoder)
        encoder = tensorflow.keras.layers.BatchNormalization()(encoder)
        # encoder = tensorflow.keras.layers.Dropout(0)(encoder) 
        encoder = tensorflow.keras.layers.Dense(200,activation='relu')(encoder) 
        # encoder = tensorflow.keras.layers.BatchNormalization()(encoder)

        self.distribution_mean = tensorflow.keras.layers.Dense(self.config.latent_dim, name='mean')(encoder)
        self.distribution_variance = tensorflow.keras.layers.Dense(self.config.latent_dim, name='log_variance')(encoder)
        self.distribution =  [self.distribution_mean, self.distribution_variance]
        latent_encoding = tensorflow.keras.layers.Lambda(self._sample_latent_features)(self.distribution)
        self.encoder_model = tensorflow.keras.Model(input_data, latent_encoding)
        self.encoder_model.summary()
        
        decoder_input = tensorflow.keras.layers.Input(shape=(self.config.latent_dim))
        decoder = tensorflow.keras.layers.Dense(200,activation='relu')(decoder_input) 
        decoder = tensorflow.keras.layers.BatchNormalization()(decoder)
        # decoder = tensorflow.keras.layers.Dropout(0.2)(decoder) 
        decoder = tensorflow.keras.layers.Dense(self.config.dense_3, activation = 'relu')(decoder)
        decoder = tensorflow.keras.layers.BatchNormalization()(decoder)
        # decoder = tensorflow.keras.layers.Dropout(0.2)(decoder) 
        decoder = tensorflow.keras.layers.Dense(self.config.dense_2, activation = 'relu')(decoder)
        decoder = tensorflow.keras.layers.BatchNormalization()(decoder)
        # decoder = tensorflow.keras.layers.Dropout(0.2)(decoder) 
        decoder = tensorflow.keras.layers.Dense(self.config.dense_1, activation = 'relu')(decoder)
        decoder = tensorflow.keras.layers.BatchNormalization()(decoder)
        # decoder = tensorflow.keras.layers.Dropout(0.2)(decoder) 
        decoder = tensorflow.keras.layers.Dense(2048, activation = 'relu')(decoder)
        decoder_output = tensorflow.keras.layers.Dense(self.config.inp_dimension) (decoder)
        
        
        self.decoder_model = tensorflow.keras.Model(decoder_input, decoder_output)
        self.decoder_model.summary()
        
        encoded = self.encoder_model(input_data)
        decoded = self.decoder_model(encoded)
        
        self.autoencoder = tensorflow.keras.models.Model(input_data, decoded)
        
    def _sample_latent_features(self,distribution):
        distribution_mean, distribution_variance = distribution
        batch_size = tensorflow.shape(distribution_variance)[0]
        random = tensorflow.keras.backend.random_normal(shape=(batch_size, tensorflow.shape(distribution_variance)[1]))
        return distribution_mean + tensorflow.exp(0.5 * distribution_variance) * random

    def _get_loss(self,distribution_mean, distribution_variance):
        
        def get_reconstruction_loss(y_true, y_pred):
            reconstruction_loss = tensorflow.keras.losses.mse(y_true, y_pred)
            reconstruction_loss_batch = tensorflow.reduce_mean(reconstruction_loss)
            return reconstruction_loss_batch
        
        def get_kl_loss(distribution_mean, distribution_variance):
            kl_loss = 1 + distribution_variance - tensorflow.square(distribution_mean) - tensorflow.exp(distribution_variance)
            kl_loss_batch = tensorflow.reduce_mean(kl_loss)
            return kl_loss_batch*(-0.5)
        
        def total_loss(y_true, y_pred):
            reconstruction_loss_batch = get_reconstruction_loss(y_true, y_pred)
            kl_loss_batch = get_kl_loss(distribution_mean, distribution_variance)
            return reconstruction_loss_batch + kl_loss_batch
        
        return total_loss
