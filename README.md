# L-d-Edutech-Hackathon
Deep Learning Framework and Machine Learning using python
......
......
......
......
......
Deep learning with Python for crack detection
.....
.....
.....
.....
.....
#GENERAL
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
#PATH PROCESS
import os
import os.path
from pathlib import Path
import glob
#IMAGE PROCESS
from PIL import Image
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.preprocessing import image
from skimage.feature import he
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from scipy.ndimage.filters import convolve
from skimage import data, io, filters
import skimage
from skimage.morphology import convex_hull_image, erosion
#SCALER & TRANSFORMATION
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras import regularizers
from sklearn.preprocessing import LabelEncoder
#ACCURACY CONTROL
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, roc_curve
from sklearn.model_selecti
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
#OPTIMIZER
from keras.optimizers import RMSprop,Adam,Optimizer,Optimizer, SGD
#MODEL LAYERS
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization,MaxPooling2D,BatchNormalization,\
                        Permute, TimeDistributed, Bidirectional,GRU, SimpleRNN, LSTM, GlobalAveragePooling2D, SeparableConv2D, ZeroPadding2D, C
                        from keras import models
from keras import layers
import tensorflow as tf
from keras.applications import VGG16,VGG19,inception_v3
from keras import backend as K
from keras.utils import plot_model
from keras.models import load_model
#SKLEARN CLASSIFIER
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble imp
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
#IGNORING WARNINGS
from warnings import filterwa
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
#IGNORING WARNINGS
from warnings import filterwarnings
filterwarnings("ignore",category=DeprecationWarning)
filterwarnings("ignore", category=FutureWarning) 
filterwarnings("ignore", category=UserWarning)
PATH,LABEL,TRANSFORMATION PROCESS
MAIN PATH
Surface_Data = Path("../input/surface-crack-detection")
JPG PATH
Surface_JPG_Path = list(Surface_Data.glob(r"*/*.jpg"))
JPG LABELS
Surface_Labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1],Surface_JPG_Path))
TO SERIES
Surface_JPG_Path_Series = pd.Series(Surface_JPG_Path,name="JPG").astype(str)
Surface_Labels_Series = pd.Series(Surface_Labels,name="CATEGORY")
TO DATAFRAME
Main_Surface_Data = pd.concat([Surface_JPG_Path_Series,Surface_Labels_Series],axis=1)
print(Main_Surface_Data.head(-1))
                                                     JPG  CATEGORY
0      ../input/surface-crack-detection/Negative/0845...  Negative
1      ../input/surface-crack-detection/Negative/1981...  Negative
2      ../input/surface-crack-detection/Negative/1691...  Negative
3      ../input/surface-crack-detection/Negative/0593...  Negative
4      ../input/surface-crack-detection/Negative/0612...  Negative
39995  ../input/surface-crack-detection/Positive/1231...  Positive
39996  ../input/surface-crack-detection/Positive/1864...  Positive
39997  ../input/surface-crack-detection/Positive/1270...  Positive
39998  ../input/surface-crack-detection/Positive/1281...  Positive

[39999 rows x 2 columns]
TO SHUFFLE
Main_Surface_Data = Main_Surface_Data.sample(frac=1).reset_index(drop=True)
print(Main_Surface_Data.head(-1))
print(Main_Surface_Data.head(-1))
                                                     JPG  CATEGORY
0      ../input/surface-crack-detection/Positive/0874...  Positive
1      ../input/surface-crack-detection/Positive/1392...  Positive
2      ../input/surface-crack-detection/Positive/1256...  Positive
3      ../input/surface-crack-
4      ../input/surface-crack-detection/Negative/1868...  Negative
...                                                  ...       ...
39994  ../input/surface-crack-detection/Positive/0517...  Positive
39995  ../input/surface-crack-detection/Negative/0490...  Negative
39996  ../input/surface-crack-detection/Negative/0065...  Negative
39997  ../input/surface-crack-detection/Negative/0586...  Negative
39998  ../input/surface-crack-detection/Negative/1976...  Negative

[39999 rows x 2 columns]
VISUALIZATION
plt.style.use("dark_background")
LABESL
Positive_Surface = Main_Surface_Data[Main_Surface_Data["CATEGORY"] == "Positive"]
Negative_Surface = Main_Surface_Data[Main_Surface_Data["CATEGORY"] == "Negative"]

Positive_Surface = Positive_Surface.reset_index()
Negative_Surface = Negativ
LABESL
Positive_Surface = Main_Surface_Data[Main_Surface_Data["CATEGORY"] == "Positive"]
Negative_Surface = Main_Surface_Data[Main_Surface_Data["CATEGORY"] == "Negative"]

Positive_Surface = Positive_Surface.reset_index()
Negative_Surface = Negative_Surface.reset_index()
VISION FUNCTION
def simple_vision(path):e_Surface.reset_index()
VISION FUNCTION
def simple_vision(path):
    figure = plt.figure(figsize=(8,8))
    
    Reading_Img = cv2.imread(path)
    Reading_Img = cv2.cvtColor(Reading_Img,cv2.COLOR_BGR2RGB)
    
    plt.xlabel(Reading_Img.shape)
    plt.ylabel(Reading_Img.size)
    plt.imshow(Reading_Img)
VISION FUNCTION
LABESL
Positive_Surface = Main_Surface_Data[Main_Surface_Data["CATEGORY"] == "Positive"]
Negative_Surface = Main_Surface_Data[Main_Surface_Data["CATEGORY"] == "Negative"]

Positive_Surface = Positive_Surface.reset_index()
Negative_Surface = Negative_Surface.reset_index()
VISION FUNCTION
def simple_vision(path):

While new technologie
VISION FUNCTION
def simple_vision(path):
    figure = plt.figure(figsize=(8,8))
    
    Reading_Img = cv2.imread(path)
    Reading_Img = cv2.cvtColor(Reading_Img,cv2.COLOR_BGR2RGB)
    
    plt.xlabel(Reading_Img.shape)
    plt.ylabel(Reading_Img.size)
    def canny_vision(path):
    figure = plt.figure(figsize=(8,8))
    
    Reading_Img = cv2.imread(path)
    Reading_Img = cv2.cvtColor(Reading_Img,cv2.COLOR_BGR2RGB)
    Canny_Img = cv2.Canny(Reading_Img,90,100)
    
    plt.xlabel(Canny_Img.shape)
    plt.ylabel(Canny_Img.size)
    plt.imshow(Canny_Img)
def threshold_vision(path):
def threshold_vision(path):
    figure = plt.figure(figsize=(8,8))
    
    Reading_Img = cv2.imread(path)
    Reading_Img = cv2.cvtColor(Reading_Img,cv2.COLOR_BGR2RGB)
    _,Threshold_Img = cv2.threshold(Reading_Img,130,255,cv2.THRESH_BINARY_INV)
    
    plt.xlabel(Threshold_Img.shape)
    plt.ylabel(Threshold_Img.size)
    plt.imshow(Threshold_Img)
    def threshold_canny(path):
    figure = plt.figure(figsize=(8,8))
    
    Reading_Img = cv2.imread(path)
    Reading_Img = cv2.cvtColor(Reading_Img,cv2.COLOR_BGR2RGB)
    _,Threshold_Img = cv2.threshold(Reading_Img,130,255,cv2.THRESH_BINARY_INV)
    Canny_Img = cv2.Canny(Threshold_Img,90,100)
    
    plt.xlabel(Canny_Img.shape)
    plt.ylabel(Canny_Img.size)
    plt.imshow(Canny_Img)
    ![image](https://user-images.githubusercontent.com/111672121/212279170-b2ad960e-f219-4777-8285-537e5f69dc76.png)
simple_vision(Main_Surface_Data["JPG"][2])
![image](https://user-images.githubusercontent.com/111672121/212279286-1ebe28f0-a5a1-41ab-bd9c-cbcaa820bd09.png)
figure,axis = plt.subplots(4,4,figsize=(10,10))

for indexing,operations in enumerate(axis.flat):
    
    Reading_Img = cv2.imread(Positive_Surface["JPG"][indexing])
    Reading_Img = cv2.cvtColor(Reading_Img,cv2.COLOR_BGR2RGB)
    
    operations.set_xlabel(Reading_Img.shape)
    operations.set_ylabel(Reading_Img.size)
    operations.imshow(Reading_Img)
    
plt.tight_layout()
plt.show()
figure,axis = plt.subplots(4,4,figsize=(10,10))

for indexing,operations in enumerate(axis.flat):
    
    Reading_Img = cv2.imread(Negative_Surface["JPG"][indexing])
    Reading_Img = cv2.cvtColor(Reading_Img,cv2.COLOR_BGR2RGB)
    
    operations.set_xlabel(Reading_Img.shape)
    operations.set_ylabel(Reading_Img.size)
    operations.imshow(Reading_Img)
    
plt.tight_layout()
plt.show()
![image](https://user-images.githubusercontent.com/111672121/212279468-e682f494-48a2-4371-ad93-1dd96df9a06c.png)

canny_vision(Main_Surface_Data["JPG"][4])
![image](https://user-images.githubusercontent.com/111672121/212279591-9b2cb747-06f2-4f25-83f2-bfb4feaf66ae.png)
canny_vision(Main_Surface_Data["JPG"][2])
![image](https://user-images.githubusercontent.com/111672121/212279704-4da23d48-25d7-40f4-903c-bc561d8a98d4.png)
figure,axis = plt.subplots(4,4,figsize=(10,10))

for indexing,operations in enumerate(axis.flat):
    
    Reading_Img = cv2.imread(Positive_Surface["JPG"][indexing])
    Reading_Img = cv2.cvtColor(Reading_Img,cv2.COLOR_BGR2RGB)
    
    Canny_Img = cv2.Canny(Reading_Img,90,100)
    
    operations.set_xlabel(Canny_Img.shape)
    operations.set_ylabel(Canny_Img.size)
    operations.imshow(Canny_Img)
    
plt.tight_layout()
plt.show()
![image](https://user-images.githubusercontent.com/111672121/212279841-2cd637dd-c57a-4cd5-ae39-39e5ce89373c.png)

![image](https://user-images.githubusercontent.com/111672121/212279890-03c11785-cfeb-454b-a24f-22f32cb89078.png)
figure,axis = plt.subplots(4,4,figsize=(10,10))

for indexing,operations in enumerate(axis.flat):
    
    Reading_Img = cv2.imread(Negative_Surface["JPG"][indexing])
    Reading_Img = cv2.cvtColor(Reading_Img,cv2.COLOR_BGR2RGB)
    
    Canny_Img = cv2.Canny(Reading_Img,90,100)
    
    operations.set_xlabel(Canny_Img.shape)
    operations.set_ylabel(Canny_Img.size)
    operations.imshow(Canny_Img)
    
plt.tight_layout()
plt.show()
![image](https://user-images.githubusercontent.com/111672121/212280088-98e8a1d5-c2a1-439f-91fb-86f0eaaf9ef1.png)
SPLITTING TRAIN AND TEST
xTrain,xTest = train_test_split(Main_Surface_Data,train_size=0.9,shuffle=True,random_state=42)
print(xTrain.shape)
print(xTest.shape)
(36000, 2)
(4000, 2)
IMAGE GENERATOR
STRUCTURE
Train_IMG_Generator = ImageDataGenerator(rescale=1./255,
                                        rotation_range=25,
                                        shear_range=0.5,
                                        zoom_range=0.5,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        brightness_range=[0.6,0.9],
                                        vertical_flip=True,
                                        validation_split=0.1)
Test_IMG_Generator = ImageDataGenerator(r
CALLBACKS
Early_Stopper = tf.keras.callbacks.EarlyStopping(monitor="loss",patience=3,mode="min")
Checkpoint_Model = tf.keras.callbacks.ModelCheckpoint(monitor="val_accuracy",
                                                      save_best_only=True,
                                                      save_weights_only=True,
                                                      filepath="./modelcheck")
                                                      Model = Sequential()

Model.add(Conv2D(32,(3,3),activation="relu",input_shape=input_dim))
Model.add(BatchNormalization())
Model.add(MaxPooling2D((2,2),strides=2))

Model.add(Conv2D(64,(3,3),activation="relu",padding="same"))
Model.add(Dropout(0.3))
Model.add(MaxPooling2D((2,2),strides=2))

Model.add(Conv2D(128,(3,3),activation="relu",padding="same"))
Model.add(Dropout(0.3))
Model.add(MaxPooling2D((2,2),strides=2))

Model.add(Conv2D(256,(3,3),activation="relu",padding="same"))
Model.add(Dropout(0.3))
Model.add(MaxPooling2D((2,2
![image](https://user-images.githubusercontent.com/111672121/212280696-00ef2163-d5b6-4b9f-8afc-ef25cb72122a.png)

plt.plot(CNN_Model.history["loss"])
plt.plot(CNN_Model.history["val_loss"])
plt.ylabel("LOSS")
plt.legend()
plt.show()
    
    plt.imshow(Reading_Img)s have changed almost every aspect of our lives, the construction field seems to be struggling to catch up. Currently, the structural condition of a building is still predominantly manually inspected. In simple terms, even nowadays when a structure needs to be inspected for any damage, an engineer will manually check all the surfaces and take a bunch of photos while keeping notes of the position of any cracks. Then a few more hours need to be spent at the office to sort all the photos and notes trying to make a meaningful report out of it. Apparently this a laborious, costly, and subjective process. On top of that, safety concerns arise since there are parts of structures with access restrictions and difficult to reach. To give you an example, the Golden Gate Bridge needs to be periodically inspected. In other words, up to very recently there would be specially trained people who would climb across this picturesque structure and check every inch of it.


Golden Gate Bridge (Photo by Free-Photos on Pixabay)
Fortunately, nowadays in cases with accessibility issues UAVs, such as drones, are deployed to take photos but still, a person would have to spend hours and hours checking each and every photo taken for signs of damage.

Here is where our work comes to revolutionize the inspection process. Artificial Intelligence takes the lead, and more specifically, Deep Learning by training our machines to be able to replace the human in the tedious task of detecting cracks on photos of structures.

There are three levels of crack detection from photos:

· The image is divided into patches and each patch is assigned a crack or non-crack label

· A rectangle is drawn around any detected crack

· Each pixel is labelled as crack or non-crack


Crack detection with image patch classiﬁcation (left), boundary box regression (mid) and pixel segmentation (right) (Dais et al, 2021)
While Deep Learning methods for crack detection have been widely studied for concrete surfaces or asphalt, little research has been done on vision-based assessment and specifically for defect detection applied to brick masonry surfaces. As part of my PhD study with my supervisors, we attempted to bridge this gap. The focus of our work is the detection of cracks on photos from masonry surfaces both on patch and pixel level. More details on our research can be found in our open access Journal paper [1]. Codes, data, and networks relevant to the implementation of the Deep Learning models can be found on my GitHub Repository [2].

Dataset preparation
The most important part of training a Deep Learning model is the data; the accuracy of a model heavily relies on the quality and amount of data. The better the representation of the real world the higher the chances of the model to be able to accurately work on real structures. Inarguably, the surface of masonry is less homogeneous and significantly noisier as compared to concrete or asphalt. Also, there are no available datasets of photos with cracks on masonry surfaces. To address the lack of data, I looked up in the Internet for any relevant photos while at the same time I took my camera and captured all the cracks in the centre of Groningen!

A common criticism over developed Deep Learning methods is that they attain remarkable results when tested on monotonous backgrounds, but their accuracy severely drops when deployed on images with complex backgrounds. Objects such as such as windows, doors, ornaments, labels, lamps, cables, vegetation etc. can be characterized as noise for the crack detection process and the network needs to learn to negate them to accurately detect cracks. Therefore, when taking photos such objects were intentionally included as well.

As a result, an extensive dataset was prepared from photos of masonry structures containing complex backgrounds and now we are ready for the next step: training the Deep Learning model.


Images of structures with and without cracks (Dais et al, 2021)

Objects that can be found on the façade of a structure (Dais et al, 2021)
Training Model
Please get prepared for the main dish!

Regarding crack detection on patch level, different state of the art CNNs pretrained on ImageNet were examined herein for their efficacy to classify images from masonry surfaces on patch level as crack or non-crack. The considered networks were: VGG16, MobileNet, MobileNetV2, InceptionV3, DenseNet121, DenseNet169, ResNet34, and ResNet50. The best results were obtained with the pretrained MobileNet, a lightweight network destined to run on computationally limited platforms. In particular, the pretrained MobileNet scored accuracy 95.3% while when no pretraining was considered the accuracy dropped to 89.0%.


Confusion matrix obtained with the MobileNet (Dais et al, 2021)
For the crack segmentation U-net and Feature Pyramid Networks, a generic pyramid representation, were considered and combined with different CNNs performing as the backbone of the encoder part of the network [3]. The CNNs used as the backbone are the networks that were previously used for patch classification. Furthermore, DeepLabv3+, DeepCrack, and FCN based on VGG16, networks that were successfully used in the literature for crack segmentation were examined as well in an extensive comparative study.

U-net-MobileNet (U-Net as base-model with MobileNet as backbone) and FPN-InceptionV3 (FPN as base-model with InceptionV3 as backbone) attained the highest F1 score, that is 79.6%. The original implementation of U-net and U-net-MobileNet without pretraining reached similar F1 score, that is 75.7% and 75.4% respectively. Therefore, using a pretrained network as the backbone boosts the F1 score by approximately 4%. Again, transfer learning seems to do the trick!

Datasets for crack segmentation are characterized by severe class imbalance i.e. the background class occupies the greatest part of photos while cracks extend over limited pixels. Due to this imbalance, if special measures are not taken, the network tends to become overconfident in predicting the background class which could lead to misclassifications of cracks and numerous false negatives. To overcome this, different loss functions were examined. The weighted cross entropy loss function, which allows the network to focus on the positive class by up-weighting the cost of a positive error, outperformed the rest.


The original image, the ground truth and the prediction with U-net-MobileNet (Dais et al, 2021)
Conclusions
With our research we showcased that the modernization of the construction sector and specifically of the inspection process is possible. Of course, these new technologies have unlimited possibilities only to be revealed with further research.

For the time being we collect additional data, further develop the crack detection process, and combine it with 3D scene reconstruction to automatically register cracks and take metric measurements.


Crack detection with 3D scene reconstruction (Image by author)
So, follow me to stay updated! 😊

👉 https://www.linkedin.com/in/dimitris-dais/

References
[1] D. Dais, İ.E. Bal, E. Smyrou, V. Sarhosis, Automatic crack classification and segmentation on masonry surfaces using convolutional neural networks and transfer learning, Automation in Construction. 125 (2021), pp. 103606. https://doi.org/10.1016/j.autcon.2021.103606.

[2] Crack detection for masonry surfaces: GitHub Repository https://github.com/dimitrisdais/crack_detection_CNN_masonry

[3] https://github.com/qubvel/segmentation_models

Data Science
Python
Computer Vision
Deep Learning
Hands On Tutorials
Thanks to Anne Bonner
288


1




Training Model
Please get prepared for the main dish!

Regarding crack detection on patch level, different state of the art CNNs pretrained on ImageNet were examined herein for their efficacy to classify images from masonry surfaces on patch level as crack or non-crack. The considered networks were: VGG16, MobileNet, MobileNetV2, InceptionV3, DenseNet121, DenseNet169, ResNet34, and ResNet50. The best results were obtained with the pretrained MobileNet, a lightweight network destined to run on computationally limited platforms. In particular, the pretrained MobileNet scored accuracy 95.3% while when no pretraining was considered the accuracy dropped to 89.0%.


Confusion matrix obtained with the MobileNet (Dais et al, 2021)
For the crack segmentation U-net and Feature Pyramid Networks, a generic pyramid representation, were considered and combined with different CNNs performing as the backbone of the encoder part of the network [3]. The CNNs used as the backbone are the networks that were previously used for patch classification. Furthermore, DeepLabv3+, DeepCrack, and FCN based on VGG16, networks that were successfully used in the literature for crack segmentation were examined as well in an extensive comparative study.

U-net-MobileNet (U-Net as base-model with MobileNet as backbone) and FPN-InceptionV3 (FPN as base-model with InceptionV3 as backbone) attained the highest F1 score, that is 79.6%. The original implementation of U-net and U-net-MobileNet without pretraining reached similar F1 score, that is 75.7% and 75.4% respectively. Therefore, using a pretrained network as the backbone boosts the F1 score by approximately 4%. Again, transfer learning seems to do the trick!

Datasets for crack segmentation are characterized by severe class imbalance i.e. the background class occupies the greatest part of photos while cracks extend over limited pixels. Due to this imbalance, if special measures are not taken, the network tends to become overconfident in predicting the background class which could lead to misclassifications of cracks and numerous false negatives. To overcome this, different loss functions were examined. The weighted cross entropy loss function, which allows the network to focus on the positive class by up-weighting the cost of a positive error, outperformed the rest.


The original image, the ground truth and the prediction with U-net-MobileNet (Dais et al, 2021)
Conclusions
With our research we showcased that the modernization of the construction sector and specifically of the inspection process is possible. Of course, these new technologies have unlimited possibilities only to be revealed with further research.

For the time being we collect additional data, further develop the crack detection process, and combine it with 3D scene reconstruction to automatically register cracks and take metric measurements.


Crack detection with 3D scene reconstruction (Image by author)
So, follow me to stay updated! 😊

👉 https://www.linkedin.com/in/dimitris-dais/

References
[1] D. Dais, İ.E. Bal, E. Smyrou, V. Sarhosis, Automatic crack classification and segmentation on masonry surfaces using convolutional neural networks and transfer learning, Automation in Construction. 125 (2021), pp. 103606. https://doi.org/10.1016/j.autcon.2021.103606.

[2] Crack detection for masonry surfaces: GitHub Repository https://github.com/dimitrisdais/crack_detection_CNN_masonry

[3] https://github.com/qubvel/segmentation_models

Data Science
Python
Computer Vision
Deep Learning
Hands On Tutorials
Thanks to Anne Bonner
288


1





Sign up for The Variable
By Towards Data Science
Every Thursday, the Variable delivers the very best of Towards Data Science: from hands-on tutorials and cutting-edge research to original features you don't want to miss. Take a look.

By signing up, you will create a Medium account if you don’t already have one. Review our Privacy Policy for more information about our privacy practices.


Get this newsletter
More from Towards Data Science
Follow
Your home for data science. A Medium publication sharing concepts, ideas and codes.

Peter Hui
Peter Hui

·Mar 4, 2021


Member-only

Power BI: Filter vs Row Context
A concept made simple (with trains). — I am what you term a “super commuter”. I used to wake up at 5 am. Leave the door at 5:30 am, catch the train and arrive at work by 8:00 am. It sounds like a nightmare. I love my job and I love it enough to do that. Then…

Data Science
5 min read

Power BI — Filter vs Row Context
Share your ideas with millions of readers.

Write on Medium
Jake
Jake

·Mar 4, 2021


Member-only

An Attack on Deep Learning
Yes, but hear me out! — Deep learning has become ubiquitous with data science and an inextricable element of machine learning. It’s shaped how humans interact with machines perhaps more so than any other advance in mathematical modeling to date. …

Data Science
8 min read

An Attack on Deep Learning
Alex Wagner
Alex Wagner

·Mar 4, 2021

Build a Multi-Layer Map Using Streamlit
Enhancing Streamlit with Plotly and Geopandas — “A map says to you, ‘Read me carefully, follow me closely, doubt me not.’ It says, ‘I am the earth in the palm of your hand. Without me, you are alone and lost.” ― Beryl Markham, West With the Night When it comes to visualizing data, everyone loves a good map. I think that is because a map makes us feel we are a part of the story being told. …

Streamlit
6 min read

Build a Multi-Layer Map Using Streamlit
Ganes Kesari
Ganes Kesari

·Mar 4, 2021


Member-only

Your AI Model is ‘Wrong’ Yet It Can Transform Your Business
Why model accuracy is overrated and how to convince your manager — For more than two decades, Netflix has been obsessed with machine learning models. In 2006, the company announced a million-dollar prize to anyone who could improve its recommendation algorithm’s accuracy by 10%. Over 40,000 teams participated in the global challenge. The competition ran for three years, and only two teams…

Artificial Intelligence
6 min read

Your AI Model is ‘Wrong’ Yet It Can Transform Your Business
Matt Przybyla
Matt Przybyla

·Mar 4, 2021


Member-only

You Should Know These 3 Data Science Visualization Tools
Here’s why… — Table of Contents Introduction Tableau Google Data Studio Pandas Profiling Summary References Introduction The use of visualizations is an important facet of data science. From the benefits in the beginning process of data science, to the end, visualizations aim to articulate complex ideas in an efficient way. Some people, like me, learn the best…

Data Science
7 min read

You Should Know These 3 Data Science Visualization Tools
Read more from Towards Data Science
Recommended from Medium
Ben Podgursky
Ben Podgursky

bunker.land — Mapping the Best Places to Wait out a Nuclear War

Bharani12
Bharani12

Use of Big Data and Analytics in the Public Sector

Mike Alatortsev
Mike Alatortsev

Satellite image data: challenges and opportunities

Alastair Majury
Alastair Majury

in

DataDrivenInvestor

Director Alastair Majury on How Companies Are Using Big Data

Fred Schenkelberg
Fred Schenkelberg

The 2 Parameter Lognormal Distribution 7 Formulas

Arief Anbiya
Arief Anbiya

in

Towards Data Science

Developing Good Twitter Data Visualizations using Matplotlib

Fred Schenkelberg
Fred Schenkelberg

ALT Allocation of Test Units
