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
While new technologies have changed almost every aspect of our lives, the construction field seems to be struggling to catch up. Currently, the structural condition of a building is still predominantly manually inspected. In simple terms, even nowadays when a structure needs to be inspected for any damage, an engineer will manually check all the surfaces and take a bunch of photos while keeping notes of the position of any cracks. Then a few more hours need to be spent at the office to sort all the photos and notes trying to make a meaningful report out of it. Apparently this a laborious, costly, and subjective process. On top of that, safety concerns arise since there are parts of structures with access restrictions and difficult to reach. To give you an example, the Golden Gate Bridge needs to be periodically inspected. In other words, up to very recently there would be specially trained people who would climb across this picturesque structure and check every inch of it.


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
