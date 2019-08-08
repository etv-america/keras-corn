# <a name="conclusions"></a> Project Outline/Findings

### Summary
The project discussed at length here represents a subsection of the end-to-end processing within the larger, overarching AgrowBot project; the purpose of this project was to create a framework which uses labelled crop data examples to readily and efficiently generate deep learning models, which can then be deployed downstream in our processing to evaluate crops live in the field. Outlined below in detail are the various components of this project, along with descriptions of their history, function, and development. For the purposes of explanation, it is assumed that the reader has a basic understanding of machine learning concepts and Python code.
## 1. Data:
Labelled data of sufficient quantity and quality is required before any deep learning model training can be done. In the particular case of this project, we require sets of at least a few thousand image examples of both healthy and diseased mature corn plants, with accurate disease labels associated with each image. The current framework we have developed can very flexibly handle new data, however, as comprehensive crop disease datasets at the present can scarcely be obtained, we have worked with just four discrete labelled datasets for the entirety of this project, which are as follow:

---
+ #### 'Sample' 
A sample control image set of four different conditions of corn (healthy, blight, rust, and tar spot, of which we only used healthy and blight for this project) taken from [Kaggle.com](https://www.kaggle.com/saroz014/plant-diseases). This set was used as a control for model training and prediction results against that of all other data in two regards:
* To ensure that the model training was behaving nominally (because of the clear distinction between the healthy and sick examples in this set, and because the pictures on this set were taken against an empty background, the model with the current layers could usually reach near perfect accuracy over this set)
* As a check to see if any issues encountered in training were the result of the model generating procedure itself or were products of errors in data manipulation via our handlers (the sample images were read in a 'un-smart' manner, by specifying each of the individual directories for the files manually, setting labels only based on folder names, and performing no image manipulation; the code cell handling the sample data was made into its present incarnation before any other sets were introduced, and the functionality of that code has not been changed since)

This set is unique in terms of content in that its images are lab data rather than field data, and is as such very easy to categorize both by human visual inspection and by a model. It is also unique in that this set was not accompanied my a csv of labels but was rather separated into names files delineating the disease(or lack of such) present.
+ #### 'Boom'
This is one of the three datasets we used for real model training, provided to use by Cornell University (the datasets and further eading on them can be found [here](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6030791/).This was the largest of the datasets used in our project, consisting of 8766 images, 1844 of which were labelled as healthy and 6922 of which were labelled as having blight. These images were taken by a camera mounted on a 5 meter boom arm, and are named apropos, as are the other two Cornell datasets. 
+ #### 'Drone'  
Another of the three Cornell sets, this was the second largest(or second smallest if you like) of the datasets used in our project, consisting of 7669 images, 2028 of which were labelled as healthy and 5641 of which were labelled as having blight. These images were taken by an aerial drone flying at a height of 6 meters.
+ #### 'Handheld'
The last of the three Cornell sets, this was the smallest of the datasets used in our project, consisting of 1787 images, 768 of which were labelled as healthy and 1019 of which were labelled as having blight. These images were taken by hand by the researchers who compiled this data. While the other two Cornell sets' content was quite similar to one another (see [prediction results](#Thisreferencedoesntexistyet)), this set proved to be semi-disparate.

---
It is worth noting that the labels for all three Cornell sets were done by hand by researchers who delineated disease by drawing a box around the area of the image which appeared to show symptoms of disease (or drawing a point at the origin, if there were none) and then saving the coordinates of this rectangle, along with an arbitrary ID number and the relevant image name, to a csv file. This raises a critical point in that **there is no way to verify the validity of the data labels**, and we are forced to put our faith in the rigor and accuracy of those researchers; to quote the researchers themselves on their [site](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6030791/), "There is no way to indicate [the] confidence of annotations...Even experts may have a hard time distinguishing between [blight] and similar-looking diseases". Nevertheless, we wrote a script which would interpret the associated coordinates to booleans (assigned an unhealthy 1 if coordinates were non-zero, assigned a healthy 0 if they were) and reformatted the csv to show only the name of each image and its associated label, for the purpose of simplifying the data to reflect only the information that was needed, as well as setting a straightforward precedent for the format of csvs for future datasets.

Each of the non-sample datasets (or *any* future labelled datasets, under the current implementation) used within our framework conceptually consist of two component documents; a file containing the images that make up that set, and a csv file, whose rows contain the names of all images in the associated image folder (in order) and a value for the disease in that image (currently a boolean representing presence of disease (1=presence of blight, 0=healthy). For a dataset called any given name, these documents must be named 'images_{name of dataset}' and 'labels_{name of dataset}.csv', respectively. If this requirement is met, the dataset may then be referenced by name throughout the model generation program with no further setup required.


### 2. Model Specifications
As the singular purpose of this framework is to produce deployable deep learning models, it is worth understanding, in detail, the conditions in which the models we have made were created. Here we will outline the layers and other configurations of the model made before any training had occurred, the hyperparameters typically used (and the experimentation yet to be done there), the training and performance of models which we have created, and the limitations we currently face.

---
+ #### Model Layers and Config
##### Figure 1, the current model layers used in our framework
![](https://github.com/timbernat/keras-corn/blob/master/Project%20Writeups/Report%20Pictures/Model_layers.png)

The model layers (shown above) are essentially exactly as they were upon creation (using [template](must-find-link!)), which appears, upon further inspection, to be a reduced VGGNet 11 [LRN] layer configuration (see figure 2), with the final two convolution layers removed and the number of neurons in each layer halved. As we lack experience in creating ground-up neural network architecture, it is unclear to us what the ideal number and configuration of layers is. 
##### Figure 2: Variations on the VGGNet Neural Netwrok architecture
![](https://github.com/timbernat/keras-corn/blob/master/Project%20Writeups/Report%20Pictures/VGGnet_model_types.png)

To better understand this, we had attempted to generate multiple successive models using fixed data and hyperparameters but with different VGGNet layer configurations (according to the Figure 2 above) to observe the effects the model configuration alone had on the quality of a model, and found that any deviation from the template layers resulted in no learning progress whatsoever occurring; the choice of the layers within the original template seemed increasingly to be far from arbitrary. Consultation with machine learning experts will be required for us to optimize this factor in model performance (see [limitations/goals](#section-not-yet-completed)).
+ #### A Discussion of Hyperparameters
[WIP]
+ #### Evaluation of Selected Models
[WIP]
+ #### Limitations and Future Goals 
[WIP]

---
