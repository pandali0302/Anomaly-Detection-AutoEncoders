import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer, MinMaxScaler
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    classification_report,
)
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
# ? Generally speaking, you will have to prioritise what you find more important. This dilemma is commonly called the "recall vs precision" trade-off. If you want to increase recall, adjust the MAD's Z-Score threshold downwards, if you want recover precision, increase it.

THRESHOLD = 3


def mad_score(points):

    m = np.median(points)
    ad = np.abs(points - m)
    mad = np.median(ad)

    return 0.6745 * ad / mad


z_scores = mad_score(mse)
outliers = z_scores > THRESHOLD

print(
    f"Detected {np.sum(outliers):,} outliers in a total of {np.size(z_scores):,} operations [{np.sum(outliers)/np.size(z_scores):.2%}]."
)


# classification_report
print(classification_report(y_test, outliers))

"""
Detected 3,523 outliers in a total of 84,807 operations [4.15%].
              precision    recall  f1-score   support

           0       1.00      0.96      0.98     84315
           1       0.11      0.78      0.19       492

    accuracy                           0.96     84807
   macro avg       0.55      0.87      0.59     84807
weighted avg       0.99      0.96      0.98     84807



"""

# ----------------------------------------------------------------
# precision_recall_curve
# ----------------------------------------------------------------
# 设置阈值并评估
precision, recall, thresholds = precision_recall_curve(y_test, z_scores)
# 可视化或选择最佳阈值
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker="o", label="Precision-Recall curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid()
plt.show()

# 设置想要达到的召回率值
desired_recall = 0.82  # 例如，想要达到90%的召回率

# 初始化变量来存储最接近的精确率和阈值
closest_recall = 0
closest_precision = 0
closest_threshold = 0

# 遍历召回率和阈值，找到最接近desired_recall的值
for i in range(len(recall)):
    if abs(recall[i] - desired_recall) < abs(closest_recall - desired_recall):
        closest_recall = recall[i]
        closest_precision = precision[i]
        closest_threshold = thresholds[i]

# 打印结果
print(f"为了达到 {desired_recall*100}% 的召回率，")
print(f"对应的精确率是：{closest_precision}")
print(f"需要设置的阈值是：{closest_threshold}")

# 输出结果
best_threshold = thresholds[np.argmax(precision + recall)]
print(f"最佳阈值: {best_threshold:.2f}")


# ----------------------------------------------------------------
# Optional - Confusion matrix for cm code
# ----------------------------------------------------------------
import itertools

# get (mis)classification
cm = confusion_matrix(y_test, outliers)
# true/false positives/negatives
(tn, fp, fn, tp) = cm.flatten()

print(
    f"""The classifications using the MAD method with threshold={THRESHOLD} are as follows:
{cm}

% of transactions labeled as fraud that were correct (precision): {tp}/({fp}+{tp}) = {tp/(fp+tp):.2%}
% of fraudulent transactions were caught succesfully (recall):    {tp}/({fn}+{tp}) = {tp/(fn+tp):.2%}"""
)


classes = [0, 1]
# create confusion matrix for cm
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
plt.show()


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


plt.subplots(figsize=(8, 8))
plt.scatter(X[labels == 0], y[labels == 0], s=1, c="g", alpha=0.3, label="Clean")
plt.scatter(X[labels == 1], y[labels == 1], s=2, c="r", alpha=0.7, label="Fraud")

# labeling
plt.legend(loc="best")
plt.title("Latent Space Representation")

# saving & displaying
plt.savefig("../../reports/figures/latent_representation_2d.png")
plt.show()
