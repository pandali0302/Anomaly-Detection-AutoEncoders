Dataset from [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) by ULB machine learning group.

When handling credit card fraud detection, class imbalance is a common and significant challenge, as fraudulent transactions typically make up only a small fraction of the total transactions. To address this issue, there are various methods to improve the model’s performance and better detect the minority class (i.e., fraudulent transactions). Below are some commonly used approaches:

    - Data Processing: Techniques such as oversampling (e.g., SMOTE), undersampling, or data augmentation can be employed to balance the dataset.

    - Model Adjustment: In algorithms like decision trees, random forests, and XGBoost, the class_weight='balanced' parameter can be used to automatically adjust the class weights.

    - Anomaly Detection Models: Since fraudulent transactions are considered anomalies, unsupervised learning methods like Auto-Encoders can be used for fraud detection. These methods do not rely on labeled fraud data but instead learn the pattern of normal transactions, identifying any transactions that deviate from these patterns as anomalies.


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