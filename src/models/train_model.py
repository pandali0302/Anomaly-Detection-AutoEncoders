import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer, MinMaxScaler
from sklearn.pipeline import Pipeline
import tensorflow as tf


RANDOM_SEED = 42
TRAINING_SAMPLE = 200000
VALIDATE_SIZE = 0.2


df = pd.read_csv("../../data/interim/01_non_skew_data.csv")
df.head()

# ? Our auto-encoder will only train on transactions that were normal. What's left over will be combined with the fraud set to form our test sample.

# ----------------------------------------------------------------
# Train/Validate/Test split
# ----------------------------------------------------------------
"""

    Training: only non-fraud
        Split into:
            Actual training of our autoencoder
            Validation of the neural network's ability to generalize
    Testing : mix of fraud and non-fraud
        Treated like new data
        Attempt to locate outliers
            Compute reconstruction loss
            Apply threshold

"""
fraud = df[df.Class == 1]
clean = df[df.Class == 0]
print(
    f"""Shape of the datasets:
    clean (rows, cols) = {clean.shape}
    fraud (rows, cols) = {fraud.shape}"""
)

# shuffle our training set
clean = clean.sample(frac=1).reset_index(drop=True)

# training set: exlusively non-fraud transactions
X_train = clean.iloc[:TRAINING_SAMPLE].drop("Class", axis=1)

# testing  set: the remaining non-fraud + all the fraud
test_set = pd.concat([clean.iloc[TRAINING_SAMPLE:], fraud]).sample(frac=1)


# train // validate - no labels since they're all clean anyway
X_train, X_validate = train_test_split(
    X_train, test_size=VALIDATE_SIZE, random_state=RANDOM_SEED
)

# manually splitting the labels from the test df
X_test, y_test = test_set.drop("Class", axis=1).values, test_set.Class.values

print(
    f"""Shape of the datasets:
    training (rows, cols) = {X_train.shape}
    validate (rows, cols) = {X_validate.shape}
    test  (rows, cols) = {X_test.shape}"""
)

# ----------------------------------------------------------------
# Normalization  and Standardization
# ----------------------------------------------------------------
# configure our pipeline
pipeline = Pipeline([("normalizer", Normalizer()), ("scaler", MinMaxScaler())])

pipeline.fit(X_train)
X_train_transformed = pipeline.transform(X_train)
X_validate_transformed = pipeline.transform(X_validate)

# ----------------------------------------------------------------
# Training the auto-encoder
# ----------------------------------------------------------------
# start TensorBorad
import tensorboard
# tensorboard --logdir=./logs
# open in browser: http://localhost:6006/

input_dim = X_train_transformed.shape[1]
BATCH_SIZE = 256
EPOCHS = 100

autoencoder = tf.keras.models.Sequential(
    [
        # deconstruct / encode
        tf.keras.layers.Dense(input_dim, activation="elu", input_shape=(input_dim,)),
        tf.keras.layers.Dense(16, activation="elu"),
        tf.keras.layers.Dense(8, activation="elu"),
        tf.keras.layers.Dense(4, activation="elu"),
        tf.keras.layers.Dense(2, activation="elu"),
        # reconstruction / decode
        tf.keras.layers.Dense(4, activation="elu"),
        tf.keras.layers.Dense(8, activation="elu"),
        tf.keras.layers.Dense(16, activation="elu"),
        tf.keras.layers.Dense(input_dim, activation="elu"),
    ]
)

autoencoder.compile(optimizer="adam", loss="mse", metrics=["acc"])

# print an overview of our model
autoencoder.summary()

# ----------------------------------------------------------------
# Callbacks
# ----------------------------------------------------------------
from datetime import datetime

# current date and time
yyyymmddHHMM = datetime.now().strftime("%Y%m%d%H%M")

# new folder for a new run
log_subdir = f"{yyyymmddHHMM}_batch{BATCH_SIZE}_layers{len(autoencoder.layers)}"

# define our early stopping
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0.0001,
    patience=10,
    verbose=1,
    mode="min",
    restore_best_weights=True,
)

save_model = tf.keras.callbacks.ModelCheckpoint(
    filepath="autoencoder_best_weights.hdf5",
    save_best_only=True,
    monitor="val_loss",
    verbose=0,
    mode="min",
)

tensorboard = tf.keras.callbacks.TensorBoard(
    f"logs/{log_subdir}", batch_size=BATCH_SIZE, update_freq="batch"
)

# callbacks argument only takes a list
cb = [early_stop, save_model, tensorboard]

# ----------------------------------------------------------------
# Training the model
# ----------------------------------------------------------------
history = autoencoder.fit(
    X_train_transformed,
    X_train_transformed,
    shuffle=True,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=cb,
    validation_data=(X_validate_transformed, X_validate_transformed),
)

# ----------------------------------------------------------------
# Reconstructions
# ----------------------------------------------------------------
# transform the test set with the pipeline fitted to the training set
X_test_transformed = pipeline.transform(X_test)

# pass the transformed test set through the autoencoder to get the reconstructed result
reconstructions = autoencoder.predict(X_test_transformed)

# calculating the mean squared error reconstruction loss per row in the numpy array
mse = np.mean(np.power(X_test_transformed - reconstructions, 2), axis=1)

clean = mse[y_test == 0]
fraud = mse[y_test == 1]

fig, ax = plt.subplots(figsize=(6, 6))

ax.hist(clean, bins=50, density=True, label="clean", alpha=0.6, color="green")
ax.hist(fraud, bins=50, density=True, label="fraud", alpha=0.6, color="red")

plt.title("(Normalized) Distribution of the Reconstruction Loss")
plt.legend()
plt.savefig("../../reports/figures/reconstruction_loss.png")
plt.show()

# ?  the fraudulent transactions clearly have a distinguishing element in their data that sets them apart from clean ones. This is a good sign that the autoencoder has learned to distinguish between the two classes.

# ----------------------------------------------------------------
# Setting a threshold for classification
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# Obtain the Latent Representations
# ----------------------------------------------------------------

encoder = tf.keras.models.Sequential(autoencoder.layers[:5])
encoder.summary()

# taking all the fraud, undersampling clean
fraud = X_test_transformed[y_test == 1]
clean = X_test_transformed[y_test == 0][:2000]

# combining arrays & building labels
features = np.append(fraud, clean, axis=0)
labels = np.append(np.ones(len(fraud)), np.zeros(len(clean)))

# getting latent space representation
latent_representation = encoder.predict(features)

print(
    f"Clean transactions downsampled from {len(X_test_transformed[y_test==0]):,} to {len(clean):,}."
)
print("Shape of latent representation:", latent_representation.shape)

X = latent_representation[:, 0]
y = latent_representation[:, 1]

# plotting
# import matplotlib as mpl
# mpl.style.use("ggplot")

plt.subplots(figsize=(8, 8))
plt.scatter(X[labels == 0], y[labels == 0], s=1, c="g", alpha=0.3, label="Clean")
plt.scatter(X[labels == 1], y[labels == 1], s=2, c="r", alpha=0.7, label="Fraud")

# labeling
plt.legend(loc="best")
plt.title("Latent Space Representation")

# saving & displaying
plt.savefig("latent_representation_2d")
plt.show()
