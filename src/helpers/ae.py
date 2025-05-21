import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

def fit_autoencoder(
    X, 
    latent_dim=5, 
    epochs=50, 
    batch_size=32, 
    validation_split=0.1, 
    learning_rate=0.001, 
    random_seed=42,
    verbose=1
):
    """
    Fits an autoencoder to input data and returns the trained encoder and latent representations.

    Args:
        X (np.ndarray): Input data array of shape (n_samples, n_features).
        latent_dim (int): Number of latent dimensions. Default is 5.
        epochs (int): Number of training epochs. Default is 50.
        batch_size (int): Training batch size. Default is 32.
        validation_split (float): Fraction of data for validation. Default is 0.1.
        learning_rate (float): Learning rate for Adam optimizer. Default is 0.001.
        random_seed (int): Seed for reproducibility. Default is 42.
        verbose (int): Verbosity level for training. Default is 1.

    Returns:
        encoder (Model): Trained Keras model for the encoder part.
        latent_features (np.ndarray): Encoded representations, shape (n_samples, latent_dim).
        scaler (StandardScaler): Fitted scaler object used for standardization.
    """
    np.random.seed(random_seed)

    n, d = X.shape

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Define the autoencoder architecture
    input_layer = Input(shape=(d,))
    encoded = Dense(128, activation='relu')(input_layer)
    encoded = Dense(64, activation='relu')(encoded)
    latent = Dense(latent_dim, activation='linear')(encoded)
    decoded = Dense(64, activation='relu')(latent)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(d, activation='linear')(decoded)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

    autoencoder.fit(
        X_scaled, X_scaled,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_split=validation_split,
        verbose=verbose
    )

    encoder = Model(inputs=input_layer, outputs=latent)
    latent_features = encoder.predict(X_scaled)

    return encoder, latent_features, scaler
