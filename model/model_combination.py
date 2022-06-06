# Internal 
from utils.utils import Utils

# External
import tensorflow 

class BaseModel(object):
    def __init__(self, config):
        self.config = config
        # self.model = None
        self.token_table = Utils().table
        

class Model_combination(BaseModel):
    """
    Constructor for the conditional generation
    Parameters
    ----------
    
    Returns
    -------
    The model's architecture for the conditional generation
    """
    def __init__(self, config):
        super(Model_combination, self).__init__(config)
        self.build_model()
        self.token_table = Utils().table
        print(len(self.token_table))
        
    def _sample_latent_features(self,distribution):
        distribution_mean, distribution_variance = distribution
        batch_size = tensorflow.shape(distribution_variance)[0]
        random = tensorflow.keras.backend.random_normal(shape=(batch_size, tensorflow.shape(distribution_variance)[1]))
        return distribution_mean + tensorflow.exp(0.5 * distribution_variance) * random

    def _get_loss(self,distribution_mean, distribution_variance):
        
        def get_reconstruction_loss(y_true, y_pred):

            # y_true = tensorflow.keras.layers.Input(shape=(self.config.inp_dimension)) 
            # y_pred = tensorflow.keras.layers.Input(shape=(self.config.inp_dimension)) 

            reconstruction_loss = tensorflow.keras.losses.mse(y_true, y_pred)
            reconstruction_loss_batch = tensorflow.reduce_mean(reconstruction_loss)
            return reconstruction_loss_batch
        
        def get_kl_loss(distribution_mean, distribution_variance):
            kl_loss = 1 + distribution_variance - tensorflow.math.square(distribution_mean) - tensorflow.math.exp(distribution_variance)
             
   
            kl_loss_batch = tensorflow.reduce_mean(kl_loss)
            kl_loss_batch = kl_loss_batch*(-0.5)
            # print(kl_loss_batch) 
            return kl_loss_batch
        
        def total_loss(y_true, y_pred):
            reconstruction_loss_batch = get_reconstruction_loss(y_true, y_pred)
            kl_loss_batch = get_kl_loss(distribution_mean, distribution_variance)

            # kl_loss_batch = tensorflow.compat.v1.convert_to_tensor(kl_loss_batch, dtype=tensorflow.float32)
            # print(kl_loss_batch)        
            # return tf.multiply(tf.convert_to_tensor(0, dtype=tf.float32, name=None),(reconstruction_loss_batch + kl_loss_batch))
            return kl_loss_batch + reconstruction_loss_batch 
        return total_loss
    

    def build_model(self):
        
        # molecular encoder
        input_mols = tensorflow.keras.layers.Input(shape=(self.config.paddSize,)) 
        encoder_mols = tensorflow.keras.layers.Embedding(len(self.token_table),self.config.embedding_dim,input_length = self.config.paddSize) (input_mols)
        encoder_mols = tensorflow.keras.layers.LSTM(self.config.units, 
                                                input_shape=(self.config.paddSize,self.config.embedding_dim),
                                                return_sequences=True) (encoder_mols)        
        encoder_mols = tensorflow.keras.layers.LSTM(self.config.units, 
                                                input_shape=(self.config.paddSize,self.config.units),
                                                return_sequences=False) (encoder_mols)  
        
        self.distribution_mean = tensorflow.keras.layers.Dense(self.config.latent_dim, name='mean')(encoder_mols)
        self.distribution_variance = tensorflow.keras.layers.Dense(self.config.latent_dim, name='log_variance')(encoder_mols)
        self.distribution =  [self.distribution_mean, self.distribution_variance]
        latent_encoding_mols = tensorflow.keras.layers.Lambda(self._sample_latent_features)(self.distribution)
        self.encoder_mol = tensorflow.keras.Model(input_mols, latent_encoding_mols)
        self.encoder_mol.summary()
        
        
        input_gep = tensorflow.keras.layers.Input(shape=(self.config.inp_dimension,))
        
        # encoder_gep = tensorflow.keras.layers.Dropout(0, seed=self.config.seed) (input_gep)
        # encoder_gep = tensorflow.keras.layers.GaussianNoise(0.1)(encoder_gep)
        
        encoder_gep = tensorflow.keras.layers.Dense(2048,activation='relu')(input_gep)
        encoder_gep = tensorflow.keras.layers.BatchNormalization()(encoder_gep)
        # encoder_gep = tensorflow.keras.layers.Dropout(0.2)(encoder_gep)        
        encoder_gep = tensorflow.keras.layers.Dense(self.config.dense_1,activation='relu')(encoder_gep)
        encoder_gep = tensorflow.keras.layers.BatchNormalization()(encoder_gep)
        # encoder_gep = tensorflow.keras.layers.Dropout(0.2)(encoder_gep) 
        encoder_gep = tensorflow.keras.layers.Dense(self.config.dense_2,activation='relu')(encoder_gep)
        encoder_gep = tensorflow.keras.layers.BatchNormalization()(encoder_gep)
        # encoder_gep = tensorflow.keras.layers.Dropout(0.2)(encoder_gep) 
        encoder_gep = tensorflow.keras.layers.Dense(self.config.dense_3,activation='relu')(encoder_gep)
        encoder_gep = tensorflow.keras.layers.BatchNormalization()(encoder_gep)
        # encoder_gep = tensorflow.keras.layers.Dropout(0.2)(encoder_gep) 
        encoder_gep = tensorflow.keras.layers.Dense(200,activation='relu')(encoder_gep) 
        # encoder_gep = tensorflow.keras.layers.Dropout(0.2)(encoder_gep) 
        # encoder_gep = tensorflow.keras.layers.BatchNormalization()(encoder_gep)
        
        self.distribution_mean_gep = tensorflow.keras.layers.Dense(self.config.latent_dim, name='mean')(encoder_gep)
        self.distribution_variance_gep = tensorflow.keras.layers.Dense(self.config.latent_dim, name='log_variance')(encoder_gep)
        self.distribution_gep =  [self.distribution_mean_gep, self.distribution_variance_gep]
        latent_encoding_gep = tensorflow.keras.layers.Lambda(self._sample_latent_features)(self.distribution_gep)
        self.encoder_gep = tensorflow.keras.Model(input_gep, latent_encoding_gep)
        self.encoder_gep.summary()
        
                
        decoder_input = tensorflow.keras.layers.Input(shape=(self.config.latent_dim))
        decoder_gep = tensorflow.keras.layers.Dense(200,activation='relu')(decoder_input) 
        decoder_gep = tensorflow.keras.layers.BatchNormalization()(decoder_gep)
        # decoder_gep = tensorflow.keras.layers.Dropout(0.2)(decoder_gep) 
        decoder_gep = tensorflow.keras.layers.Dense(self.config.dense_3, activation = 'relu')(decoder_gep)
        decoder_gep = tensorflow.keras.layers.BatchNormalization()(decoder_gep)
        # decoder_gep = tensorflow.keras.layers.Dropout(0.2)(decoder_gep) 
        decoder_gep = tensorflow.keras.layers.Dense(self.config.dense_2, activation = 'relu')(decoder_gep)
        decoder_gep = tensorflow.keras.layers.BatchNormalization()(decoder_gep)
        # decoder_gep = tensorflow.keras.layers.Dropout(0.2)(decoder_gep) 
        decoder_gep = tensorflow.keras.layers.Dense(self.config.dense_1, activation = 'relu')(decoder_gep)
        decoder_gep = tensorflow.keras.layers.BatchNormalization()(decoder_gep)
        # decoder_gep = tensorflow.keras.layers.Dropout(0.2)(decoder_gep) 
        decoder_gep = tensorflow.keras.layers.Dense(2048, activation = 'relu')(decoder_gep)
        decoder_output = tensorflow.keras.layers.Dense(self.config.inp_dimension,activation = 'linear') (decoder_gep)

        
        self.decoder_gep = tensorflow.keras.Model(decoder_input, decoder_output)
        self.decoder_gep.summary()
            
        encoded_mols = self.encoder_mol(input_mols)
        encoded_gep = self.encoder_gep(input_gep)
        
        # Different strategies to combine the molecular and genomic latent spaces
        
        if self.config.joint_strategy == 'add':
            latent_space = tensorflow.keras.layers.Add()([encoded_mols, encoded_gep])
        elif self.config.joint_strategy == 'average':
            latent_space = tensorflow.keras.layers.Average()([encoded_mols, encoded_gep])
        elif self.config.joint_strategy == 'multiply':
            latent_space = tensorflow.keras.layers.Multiply()([encoded_mols, encoded_gep])
        
        # print(latent_space)
     
        decoded_gep = self.decoder_gep(latent_space)
        
        # self.autoencoder_gep = tensorflow.keras.models.Model(input_gep, decoded_gep)
        self.model = tensorflow.keras.Model(inputs=[input_mols, input_gep],outputs=decoded_gep)
        self.model.summary()
        
        self.encoder_mol.trainable = False
        self.encoder_gep.trainable = False
        self.reconstruction_loss = tensorflow.keras.losses.mse(input_gep, decoded_gep)
        self.reconstruction_loss_batch = tensorflow.reduce_mean(self.reconstruction_loss)
        self.kl_loss = 1 + self.distribution_variance_gep - tensorflow.math.square(self.distribution_mean_gep) - tensorflow.math.exp(self.distribution_variance_gep)
        # print(self.reconstruction_loss_batch)
        # print(self.kl_loss)
        self.kl_loss = tensorflow.reduce_mean(self.kl_loss)
        # print(self.kl_loss)
        self.kl_loss = self.kl_loss*(-0.5)
        # print(self.kl_loss)
        self.vae_loss = self.kl_loss + self.reconstruction_loss_batch
        # print(self.vae_loss)
        self.model.add_loss(self.vae_loss)
        self.model.compile(optimizer='adam')
        # self.model.compile(loss=self._get_loss(self.distribution_mean_gep, self.distribution_variance_gep),  optimizer='adam', experimental_run_tf_function=False)

