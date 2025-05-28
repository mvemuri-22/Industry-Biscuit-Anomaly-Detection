# Industrial Biscuit Defect Detection

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Data Preparation](#data-preparation)
4. [Model Training](#model-training)
    * [Autoencoder](#autoencoder)
    * [Variational Autoencoder (VAE)](#variational-autoencoder-vae)
    * [Generative Adversarial Network (GAN)](#generative-adversarial-network-gan)
    * [Vision Transformer (ViT)](#vision-transformer-vit-vae)
5. [Final Model](#final-model-and-results)
6. [Future Work](#future-work)

## Introduction

This project aims to develop a robust anomaly detection system for industrial biscuit quality control. We will explore and implement various deep learning models, including Variational Autoencoders (VAEs) and Generative Adversarial Networks (GANs), to identify defective biscuits based on their visual appearance. The primary goal is to distinguish between non-defective ("ok") and defective ("nok") biscuits by learning the distribution of normal biscuit features. This is particularly crucial in industrial settings where the cost of a false negative (selling a defective product) can be significantly higher than a false positive (discarding a non-defective product). Selling defective biscuits can lead to customer dissatisfaction, brand damage, recalls, and in cases of safety concerns (e.g., unknown foreign objects or harmful colorations), severe health risks and legal liabilities. Therefore, our anomaly detection system prioritizes minimizing the risk of defective products reaching the market. This README provides an overview of our methodology, the models we plan to use, the dataset, and our approach to data preparation, model training, and evaluation.

## Dataset

We will be utilizing a specialized dataset comprising 1,225 images of Tarallini biscuits sourced from an industrial plant. Each image has a resolution of 256x256 pixels. The dataset categorizes biscuits into four types: "no defect," "not complete," "strange object," and "color defect." To ensure a comprehensive view for analysis, each biscuit is captured from four different angles. This rich dataset closely mimics real-world quality control scenarios, making it highly suitable for developing an effective defect detection model in an industrial setting.

The dataset was obtained by running the provided code/notebook from the Kaggle link. It is organized into `train`, `test`, and `val` (validation) folders. Within each of these folders, there are subfolders named `ok` (for non-defective biscuits) and `nok` (for defective biscuits), allowing for clear separation of normal and anomalous samples.

Below are some example images from the training dataset for both non-defective and defective cookies:

![image](https://github.com/user-attachments/assets/7a4a0bd1-0daf-4b7a-82b1-9e9ea1e0dc78)


![image](https://github.com/user-attachments/assets/651b780c-9609-4917-959d-86345b3dd50c)


For more details, the dataset can be accessed at: [Industry Biscuit Cookie Dataset](https://www.kaggle.com/datasets/imonbilk/industry-biscuit-cookie-dataset/data)

## Data Preparation

The data preparation phase involves several crucial steps to ensure the images are in a suitable format for model training.

1. Loading Images: Images are loaded from the specified directories (`train/ok` and `train/nok`). The `preprocess_image` function handles loading, resizing, and normalization.  
2. Resizing and Normalization: All images are resized to a uniform dimension of `256x256` pixels and normalized to a pixel range of by dividing by 255\. This standardization is critical for consistent model input.  
3. **Data Splitting**: To prevent data leakage, especially since each cookie is captured from 4 different angles (resulting in 4 consecutive images of the same cookie rotated), the splitting process is performed by grouping images in sets of 4 before distributing the training set images into training and validation sets. The test set is kept as is from the data extraction output given by the data source. All datasets (`X_train`, `X_val`, `X_test`) are converted to `float32` and normalized to the range.

## **Model Training**

### **Autoencoder**

To set a baseline with a non-bayesian technique we used a convolutional autoencoder to detect anomalies. We train this model on images of normal biscuits to learn how to compact the representation of a real cookie. For predictions we then flag images where the reconstruction error is high as a potential anomaly. 

```python
from tensorflow.keras import layers, models

def build_autoencoder(input_shape):
    input_img = layers.Input(shape=input_shape)

    # Encoder
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)  # Compressed representation

    # Decoder
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = models.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder
```


When running this across our test set we are able to get a test accuracy of 81% with a recall of 56% on anomalies. During the development process we adjusted thresholds to prioritize recall of anomalies, but this came at the cost of wasting too many good cookies. As a result we did a closer examination of the missed anomalies.

             precision    recall  f1-score   support

           0       0.69      0.96      0.80       200

           1       0.93      0.56      0.70       200

    accuracy                           0.76       400

*Examination of Errors*

As discussed above, our dataset contains three kinds of anomalies: colors, shape, and defects. The two we are most concerned about from a business perspective are colors and defects. There are biscuits in the dataset that have red and black coloring that could indicate a safety hazard. For defects we have some cookies with a nail or metal screw in it that could be very dangerous.

When we look at our true positives , we do a good job catching these kinds of dangerous anomalies.

**True Positive Images:**

![image](https://github.com/user-attachments/assets/f5434de1-7630-4978-984c-bdf6fdd4ba95)


Where we miss the mark are shape defects, which in our analysis are less risky to the business.

**False Negative Images:**

![image](https://github.com/user-attachments/assets/45a69727-7574-46df-88b1-61fe04838518)


This is a promising start, with the autoencoder but we now move forward to building out a latent space for these images with a VAE.

### **Variational Autoencoder (VAE)**

We also tested the Variational Autoencoder (VAE) for binary classification of the cookies: defect vs non-defect. VAE is trained on the "normal" (non-defective) biscuit images and uses that to learn a latent representation. Anomalies are then identified by measuring the reconstruction error of test samples – if the error exceeds a certain threshold, it would be considered an anomaly

* The VAE consists of an encoder and a decoder network. Below is the configuration of our architecture:

```python
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Lambda, Reshape, Conv2DTranspose
from tensorflow.keras import backend as K

# Input: Images of size (64, 64, 3)
encoder_input = Input(shape=(64, 64, 3))
x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoder_input)
x = Conv2D(64, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
x = Flatten()(x)
x = Dense(64, activation='relu')(x)

# Latent space parameters
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# Sampling Layer (Reparameterization Trick)
# z = mu + sigma * epsilon, where epsilon ~ N(0, I)
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.)
    return z_mean + K.exp(z_log_var / 2) * epsilon

z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# Decoder Architecture
# Input: Sampled latent vector z of size (latent_dim,)
decoder_input = Input(shape=(latent_dim,))
x = Dense(8 * 8 * 128, activation='relu')(decoder_input)
x = Reshape((8, 8, 128))(x)
x = Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
x = Conv2DTranspose(64, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
x = Conv2DTranspose(32, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
decoder_output = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
```


* Loss Function: The VAE loss function combines two components:  
  1. Reconstruction Loss: Measures how well the VAE reconstructs the input image. We use binary cross-entropy, which is suitable for images normalized to \[0,1\].  
     * Lreconstruction​=−∑i​(xi​log(x^i​)+(1−xi​)log(1−x^i​))  
  2. KL Divergence Loss: Regularizes the latent space by forcing the learned distribution to be close to a standard normal distribution. This ensures a continuous and well-structured latent space, preventing overfitting and encouraging meaningful latent representations.  
     * LKL​=−0.5∑i​(1+log(σi2​)−μi2​−σi2​) The total VAE loss is the sum of these two components.

Training Process: As mentioned, VAE is trained exclusively on non-defective cookie images (`X_train_ok`). The model learns to encode these images into a latent space and then decode them back to their original form. During training, the reconstruction error is minimized, and the latent space is regularized. The model is compiled with the Adam optimizer and the custom VAE loss function as mentioned in the previous part. We trained the model for `50` epochs with a `batch_size` of `32`.

#### VAE Results and Analysis

##### Reconstruction Visualization

After training, the VAE's ability to reconstruct images was evaluated. The model was able to reconstruct non-defective cookie images . For defective images however, we see some flaws in the reconstruction as the model attempts to reconstruct them as non-defective cookies.

![image](https://github.com/user-attachments/assets/f618c325-e18b-4817-ab54-382ac341f628)


##### Anomaly Detection Thresholding

We set the optimal threshold for the Anomaly detection by selecting the threshold that obtains the F1-Score on the validation set. We further tuned the threshold to achieve the optimal anomaly detection performance based on using standard metrics including accuracy, precision, recall, and F1-score.

* Optimal Threshold: 0.5047  
* Results on Test Set:   
  * Accuracy: 0.79  
  * Precision: 0.72  
  * Recall: 0.95  
  * F1-Score: 0.82

##### Clustering Defective Images

In addition to the simple binary defect detection, we decided to further explore and understand the different types of defects. From the dataset source, we were informed that there were 3 different types of defects: not complete, strange object  and color defect. This task is important from a business standpoint since certain defects could still be acceptable (i.e. hape defects) which could still be sold to customers compared to other defects which could be detrimental to the business if sold to customers (i.e. unknown objects or coloring in the cookies). To identify the different defects, we applied K-Means clustering on the latent space representations of the predicted defects from the test samples (K=3). We obtained the following distribution:

* Cluster 0: 76 images  
* Cluster 1: 76 images  
* Cluster 2: 48 images

Visualization on a 2D plot shows that Cluster 2 is very distinct, while Cluster 0 and Cluster 1 were less distinguishable.

![image](https://github.com/user-attachments/assets/f325604d-d1ac-4670-9e5c-be60363b6360)


After further examination we see that Cluster 2 actually corresponds to cookies that have shape defects. In fact, we see that out of 48 cookies in Cluster 2, 44 are shape defects and the other 4 are color defects. This shows a promising result that we were able to identify the difference between “acceptable” defects vs unsellable defects.

*Original Cookie Images in Cluster 2*

![image](https://github.com/user-attachments/assets/6a9cc0c7-1d4d-4043-bfd8-229efe1efe78)


##### VAE on Defects Only

In attempts to explore the possibility of obtaining a better clustering result on the defective cookies, we experiment by training a separate VAE model exclusively on the defective images. The motivation behind this was to explore if a VAE trained specifically on defects could learn a better latent space for classifying different defect types. However, our results indicated that this approach did not yield a better result for defect type classification. Our conclusion from this was that while it was able to build a good latent space for defect type reconstruction, it didn't work as effectively for defect classification.

### **Vision Transformer (ViT) VAE**

To push the boundaries of representation learning, we experimented with a Vision Transformer-based Variational Autoencoder (ViT-VAE). The core idea is to leverage the pretrained representational power of a Vision Transformer (ViT) to encode visual features more robustly than traditional CNN encoders. This approach uses a pretrained ViT-Base model as the encoder and a transposed convolutional network as the decoder to reconstruct the input image from a sampled latent space.

#### ViT-VAE Architecture

* **Encoder**: We use the ViT-Base (google/vit-base-patch16-224) pretrained model from Hugging Face Transformers.   
  * This is an encoder only ViT.  
  * The \[CLS\] token is excluded, and only the patch embeddings are used.   
  * These are reshaped into a 2D feature map and passed through a final convolutional layer to output mean and log-variance vectors for the latent space.

* **Latent Space**: The VAE latent space follows the standard Gaussian reparameterization trick:  
$z = \mu + \sigma \cdot \epsilon \quad \text{where} \quad \epsilon \sim \mathcal{N}(0, I)$

* **Decoder**: A 4-layer transposed convolutional decoder reconstructs the image from the latent vector z. Batch normalization and ReLU activations are applied in intermediate layers.

* **Loss Function**: Combines reconstruction loss (binary cross-entropy) and KL divergence.


#### Training Details

* Input Size: 224×224×3 (resized from 256×256)  
* Latent Dimension (z\_dim): Tuned to balance complexity and regularization  
* Training Epochs: 100  
* Batch Size: 32  
* Optimization: Adam optimizer  
* KL Divergence Weight (β): 1.0

#### ViT-VAE Results and Analysis

Despite the architectural sophistication and excellent reconstructions, the ViT-VAE showed poor test set classification performance. **One likely cause is that the pretrained ViT encoder was too good at reconstructing even defective images, including those with nails or unusual coloring**. This means that even anomalies were reconstructed well, leading to low reconstruction error and consequently poor anomaly discrimination.

1. Threshold Tuning (based on validation set reconstruction error):

* Best Threshold: 70,077.57  
* Validation Set Performance:  
  * Precision: **0.5000**  
  * Recall: 1.0000  
  * F1 Score: 0.6667  
  * AUC: 0.5058

2. Test Set Performance:

* Precision: **0.4898**  
* Recall: 0.9600  
* F1 Score: 0.6486  
* AUC: 0.6091

3. Reconstruction Performance:

![image](https://github.com/user-attachments/assets/7f8c1145-42e0-4de5-a759-0b4c9eb58ba1)

 *Figure: ViT-VAE reconstructions on test set. Even severe anomalies are reconstructed clearly.*

#### Latent Space Analysis

The t-SNE visualization of the latent space reveals slightly distinct but overlapping clusters for defective and non-defective samples. **Defective cookies tend to reside near the boundary of the normal cluster, which confirms that the latent space does learn some distinction**. However, the boundaries are not sharp, possibly due to the ViT encoder’s generalization ability, causing defective features to be encoded similarly to normal ones.



![image](https://github.com/user-attachments/assets/ff970b91-392f-4359-9e0f-7d0fbea9c82b)

 *Figure: t-SNE visualization of ViT-VAE latent space. Anomalies (red) are loosely grouped near normal samples (green).*

#### Conclusion

While the ViT-VAE demonstrates strong reconstructive capabilities and potential for rich feature extraction, its overly smart encoder and expressive decoder limits its effectiveness in anomaly detection. This model contributes to our understanding of the tradeoff between representational quality and anomaly separability in latent spaces learned via deep generative models.

### **Generative Adversarial Network (GAN)**

We also trained a Deep Convolutional Generative Adversarial Network (DCGAN) to identify anomalous biscuits. 
Architecture: DCGAN consists of a Generator and a Discriminator network and our model has the following architecture:

```python
class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_size=64, channels=3):
        super(Generator, self).__init__()
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img



class Discriminator(nn.Module):
    def __init__(self, img_size=64, channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [
                nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25)
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128)
        )

        ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity
```

#### Training Process: 
The GAN is trained iteratively, alternating between updating the Discriminator and the Generator. Similar to other models we train the DCGAN model on normal biscuits and find which images have high reconstruction errors. 
* Discriminator Training: The Discriminator is trained to distinguish between real images (from the dataset) and fake images (generated by the Generator). It aims to output a high probability for real images and a low probability for fake images.
* Generator Training: The Generator is trained to produce images that can fool the Discriminator. Its loss function combines the adversarial loss (how well it fools the discriminator) and the feature loss (how similar its generated images are to real images in terms of extracted features).
* Optimizers: Both the Generator and Discriminator are optimized using Adam with a learning rate of 0.0002 and betas (0.5, 0.999).
* Epochs: The model is trained for 100 epochs.

#### DCGAN Results and Anomaly Detection
Anomaly detection using the DCGAN is performed by evaluating the reconstruction error. The generator, trained only on normal biscuits, struggles to perfectly reconstruct anomalous images.

* Optimal Threshold: By leveraging Youden’s J statistic, an optimal threshold of 0.0468 was determined for anomaly detection.

* Results on Test Set: Using the optimal threshold, the DCGAN achieved the following performance on the test set:
   * Accuracy: 0.8933
   * Precision (Anomaly): 0.9200
   * Recall (Anomaly): 0.8583
   * F1-Score (Anomaly): 0.8889

We see below what the reconstruction errors looks like for normal vs anomalous cookies: 

![image](https://github.com/user-attachments/assets/acc34772-7af4-4a0c-8046-72c6d42a3e39)

* Confusion Matrix:

![image](https://github.com/user-attachments/assets/b11137f1-6e97-49c9-90af-4eb4db8b5e85)


## Final Model and Results

Given our results, our best model for identifying anomalies is the DCGAN. The overall accuracy of the model is the best of the group and it also does a great job in terms of recall of anomalies (85%). The GAN is able to give sharper, more realistic images which allows it to see some of the fine grain details. Models like VAE and autoencoder struggled to identify shape defects for example, whereas DCGAN was able to capture more of these without giving up precision. A DCGAN is also less likely to explain away any anomalies by reconstructing them well.

From a business perspective we are striving to minimize false negatives because they represent a large risk to the companies success. When we look at some of the color defects or cookies with metal items in them, if served this could result in a lawsuit. On the positive, all models do a good job of capturing these major red flags. But models like autoencoder and VAE, may result in more false positives which means wasted cookies. DCGAN outperforms in terms of precision and recall meaning we are avoiding risk of serving bad cookies while also producing enough cookies for the business to be successful.

## Future Work

* **Achieving Production Readiness:** To successfully deploy such a model in a real industrial setting, we must achieve significantly improved metrics, particularly higher accuracy and precision, ideally striving for 100% to minimize the risk of false negatives. This will involve further model refinement and rigorous testing under diverse real-world conditions.  
* **Expanding Scope and Generalizability to other Fields:** Our anomaly detection approach is highly adaptable and can be applied beyond the cookie industry. We aim to explore its use in other fields, such as medical imaging analysis, where identifying anomalies (e.g., tumors or abnormalities in X-rays or MRI scans) is crucial for diagnosis and treatment.
