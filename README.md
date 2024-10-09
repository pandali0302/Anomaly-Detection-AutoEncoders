Dataset from [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) by ULB machine learning group.

We are using Auto-Encoder for **Anomaly Detection**: Auto-encoders can be used to learn the normal behavior of the data and detect anomalies by identifying instances that cannot be well reconstructed (i.e., have high reconstruction error)

Auto-Encoders (AEs) are a type of unsupervised learning model primarily used for tasks like dimensionality reduction, feature learning, and anomaly detection. They are neural networks trained to reconstruct input data, making them effective at capturing key features of the data in a compressed representation.

![Autoencoder: schema](https://blog.keras.io/img/ae/autoencoder_schema.jpg)

In **unsupervised learning with auto-encoders**, the model learns to reconstruct input data from a compressed, low-dimensional "bottleneck" representation, which captures important underlying features. This technique is especially useful when you don’t have labeled data and are interested in detecting anomalies or learning meaningful features from the data.


### Components of an Auto-Encoder

1. **Encoder**: Maps the input data xxx to a latent representation zzz in a compressed form.
2. **Latent Space**: The bottleneck layer, which holds the compressed version of the input. This lower-dimensional space forces the network to focus on the most important patterns.
3. **Decoder**: Attempts to reconstruct the original input x^\hat{x}x^ from the latent representation zzz.

The network is trained by minimizing the reconstruction error, which is the difference between the input xxx and the reconstructed output x^\hat{x}x^.



### Steps for Unsupervised Learning with Auto-Encoders

1. **Data Preparation**: Ensure your data is scaled appropriately (commonly between 0 and 1 for auto-encoders).
2. **Build Auto-Encoder Model**: Construct a neural network with an encoder and decoder. The size of the bottleneck layer will determine the dimensionality of the latent space.
3. **Train the Auto-Encoder**: Use the input data as both the input and the target. Train the model to minimize the reconstruction error.
4. **Evaluate Reconstruction Error**: For anomaly detection, compute the reconstruction error for each data point. Data points with high reconstruction error are likely to be anomalies.
5. **Optimize Threshold for Anomaly Detection**: Determine a threshold for the reconstruction error to classify an instance as an anomaly.