# <a name="conclusions"></a> Project Outline/Findings

### Summary
The project discussed at length here represents a subsection of the end-to-end processing within the larger, overarching AgrowBot project; the purpose of this project was to create a framework which uses labelled crop data examples to readily and efficiently generate deep learning models, which can then be deployed in our downstream processing to evaluate crops for disease live in the field. Outlined in detail in this report are the various technical specifications used within this project and all observations we made during development. For the purposes of explanation, it is assumed that the reader has a basic understanding of machine learning concepts and Python code.
## <a name="data"></a> Data:
Labelled data of sufficient quantity and quality is required before any deep learning model training can be done. In the particular case of this project, we require sets of at least a few thousand image examples of both healthy and diseased mature corn plants, with accurate disease labels associated with each image. The current framework we have developed can flexibly interprat and train over an arbitrary amount of new data; however, as comprehensive crop disease datasets are scarce and difficult to obtain at the present, we have worked with just four discrete labelled datasets for the entirety of this project, which are as follow:

---
---
+ #### <a name="sample"></a> 1. 'Sample' 
A sample control image set of corn in four different states of disease(healthy, blight, rust, and tar spot, of which we only used healthy and blight for this project) taken from [Kaggle.com](https://www.kaggle.com/saroz014/plant-diseases). This set was used early on as a control for the performance of all other models in two regards:
* To ensure that the model training was behaving nominally. Because of the clear distinction between the healthy and sick examples in this set, and because the pictures on this set were taken against an empty, noiseless background, the model produced from training over these images, with our [standard model configuration](#models), could always reach above 95% evaluation accuracy; any significant deviation from this performance was a sign that errors had occurred in the generation procedure.
* To see whether poor evaluation performance was the result of the model generation procedure or of errors in data manipulation by our handlers. The sample images were read in a 'un-smart' manner, by specifying each of the individual directories for the files manually, setting labels only based on folder names, and performing no feature or label manipulation; the code cell handling the sample data was made into its present incarnation before the reading of any other datasets was introduced, and the functionality of that code has not been changed since; hence, any errors manifesting in both other model and sample model performance were the result of errors in the generation procedure rather than in the data reading.

This set is unique in terms of content in that its images are lab data rather than field data, and is as such very easy to categorize both by human visual inspection and by a model. It is also unique in that this image set was not accompanied my a csv of labels but was rather separated into named files delineating the disease (or lack of such) present within that folder.
+ #### <a name="boom"></a> 2.'Boom'
This is one of the three datasets we used for training our applied model, with the data being provided for our use by Cornell University (these datasets and further reading on them can be found [here](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6030791/). This was the largest of the datasets used in our project, consisting of 8766 images, 1844 of which were labelled as healthy and 6922 of which were labelled as having blight. These images were taken by a camera mounted on a 5 meter boom arm, and are named apropos, as are the other two Cornell datasets. 
+ #### <a name="drone"></a> 3. 'Drone'  
Another of the three Cornell datasets; this was the second largest(or second smallest if you like) of the datasets used in our project, consisting of 7669 images, 2028 of which were labelled as healthy and 5641 of which were labelled as having blight. These images were taken by an aerial drone flying at a height of 6 meters.
+ #### <a name="handheld"></a> 4. 'Handheld'
The last of the three Cornell sets; this was the smallest of the datasets used in our project, consisting of 1787 images, 768 of which were labelled as healthy and 1019 of which were labelled as having blight. These images were taken by hand by the researchers who compiled this data. While the other two Cornell sets' content was quite similar to one another, this set proved to be slightly disparate from the others (see in [batch size](#batch-size)) for detail.

---
---
It is worth noting that the labels for all three Cornell sets were done by hand by researchers who delineated disease by drawing a box around the area of the image which appeared to show symptoms of disease (or drawing a point at the origin, if there were none) and then saving the coordinates of this rectangle, along with an arbitrary ID number and the relevant image name, to a csv file. This raises a critical point in that **there is no way to verify the validity of the data labels**, and we are forced to put our faith in the rigor and accuracy of those researchers; to quote the researchers themselves on their [site](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6030791/), <a name="data-ambiguity"></a> "There is no way to indicate [the] confidence of annotations...Even experts may have a hard time distinguishing between [blight] and similar-looking diseases".

Notwithstanding this, we wrote a script which would convert the associated coordinates to booleans (assigned an 1 if coordinates were non-zero(crop was diseased), or 0 if they were 0(crop was healthy)) and reformatted the csv to show only the name of each image and its associated label, for the purpose of simplifying the data to reflect only the information that was needed, as well as setting a straightforward precedent for the format of csvs for future datasets.

Each of the non-sample datasets (or *any* future labelled datasets, under the current implementation) used within our framework conceptually consist of two component documents; a file containing the images that make up that set, and a csv file, whose rows contain the names of all images in the associated image folder (in order) and a value for the disease in that image (currently a boolean representing presence of disease (1=presence of blight, 0=healthy). For a dataset called any given name, these documents must be named 'images_{name of dataset}' and 'labels_{name of dataset}.csv', respectively. If this requirement is met, the dataset may then be referenced by name throughout the model generation program with no further setup required.


## <a name="models"></a> Models
Beyond the data used for training models, there are a number of configurations and parameters within the model training procedure that directly shape the nature of the models generated by this framework. Here we will discuss those which are the most significant/imapctful and provide our observations of their effects on model performance and, where applicable, areas which require further research and optimisation.

---
+ ### <a name="model-layers"></a> 1. Model Layers and Pre-config
##### Figure 1, the current model layers used in our framework
![](https://github.com/etv-america/keras-corn/blob/master/Project%20Writeups/Report%20Pictures/Model_layers.png)

The model layers as they exist at the present (shown above) remain almost exactly as they were upon creation (using [template](must-find-link!)), which appears, upon further inspection, to be a reduced VGGNet 11 [LRN] layer configuration (see figure 2 below), with the final two convolution layers removed and the number of neurons in each layer halved. As we do not have expertise in creating ground-up neural network architecture, it is unclear to us what the ideal number and configuration of layers is to facilitate generation of optimally performing models. 
##### Figure 2: Variations on the VGGNet Neural Network architecture
![](https://github.com/etv-america/keras-corn/blob/master/Project%20Writeups/Report%20Pictures/VGGnet_model_types.png)

To better understand this, we had attempted to generate several successive models using a fixed dataset and hyperparameters with varied VGGNet layer configurations (according to the figure 2 above) to observe the effects the model configuration alone had on the quality of a model, and found that any deviation from the template layers resulted in no learning progress whatsoever occurring (with the exception of halving the number of neurons in the dense layer from 512 to 256; this slightly decreased overfitting); the choice of the layers within the original template seemed increasingly to be far from arbitrary. Consultation with machine learning experts will be required for us to optimize our framework in this regard.
+ ### <a name="model-detail"></a> 2. Model Specifications/Optimisation
Because the majority of our limited project time was spent developing the model generation framework itself and validating that it was functioning consistently and as intended, and because of the time and amount of computing required to generate the quantity and quality of models needed to be conducive to analysis, we were only able to very briefly experiment with optimization of models generated; much of the further development of this project lies in this area. The [performance of *specific* models](#selected-models) that we produced is discussed in more detail later in this report; here we outline the general trends in performance we have observed, as well as all optimisation and model tuning configurations used within this project and their effects on model generation and evaluation performance. 

---
---
* #### <a name="bs"></a> Batch Size: 
The size of our data being only in the order of a few thousands, we opted for using 'mini-batches' of size ranging from 2-32, as is used even in larger neural networks, to better accomodate the variation in the image data we used. We (somewhat arbitrarily) always kept this to powers of two in our experimentation, and we found that 32 appears the give the best actual model performance, typically 70-80% accurate over the data it was trained over (models trained over drone perform similarly accurately over the boom set, and vice versa), and between 60-70% accurate over more disparate sets (the handheld set in our particular case), with smaller batch sizes(2-4) giving ostensibly identical accuracy results while training but performancing extrememly poorly when actually evaluating over data (these typically worse than just guessing and often will predict everything as being either sick or healthy). Across all batch sizes, however, our models always showed signs of data overfitting (see [here}(#nr-epochs) for more detail).
* #### <a name="lr"></a> Learning Rate: 
7*10^-4 is the rate we used most commonly in the successful models that we preserved; in general, our learning rates for this project lay in the order of 10^-4, as across all the data we had available, learning progress seemed to dramatically drop off above 10^-3 and perform well but far to slowly near and below 10^-5. More granular testing over this parameter was not done, as we were already producing quite favorable results with the learning rate in the order of 10^-4 and would have lost a great deal of time in exchange for little improvement had we done so.
* #### <a name="nr-epochs"></a> Number of Epochs: 
This proved to be the most active area of testing, observation, and refinement within the optimisation portion of this project; in order to rigorously and definitively identify the impact of various parameters, a single parameter had to be chosen and varied across several training sessions while all other settings stayed the same; this is a very time consuming process, and as such, much of the observation and tweaking in this project was done on the number of epochs, with the other hyperparameters and settings listed here being kept static at what we found to be the best, if not definitely optimal, values. All models exhibited logarithmically increasing growth in training accuracy, and linearly increasing growth in validation accuracy, often at a very gradual or even zero rate; this is a clear sign of overfitting to our training data, and is likely due to the ambiguity of the image sets we used, as described [prior](#data-ambiguity). From examination of the larger models we created, the training performance beyond ~500 epochs grows so slowly, compared the to the growth prior to that point, that this could be considered a cutoff point for time efficiency in producing high-quality models going forward.
##### Figure 3: Performance and Loss graphs for our 800-epoch and 2048-epoch models
![](https://github.com/etv-america/keras-corn/blob/master/Project%20Writeups/Report%20Pictures/800%E2%81%842048%20perf.png)
* #### <a name="optimizer"></a> Optimizer Settings: 
##### Figure 5: the model optimizer configuration line in our code
![](https://github.com/etv-america/keras-corn/blob/master/Project%20Writeups/Report%20Pictures/opti-config.png)
While these do not describe a hyperparameters, they are nevertheless of equal relevance to performance as the other hyperparameters discussed here. Our model seemed to perform best with using the 'Nadam' optimizer implementation with Keras, performing farily similarly with 'Adagrad' and slightly more poorly using 'RMS Propagation'; 'Stochastic Gradient Descent' proved poorly suited due to the small size of our data for that implementation, and the other variations of the 'Adam' optimizer were not tested thoroughly enough to make any definitive conclusions at the present. Because we are currently operating within a binary classification paradigm, there were no loss functions available beyond the 'binary crossentropy' function we are currently using that would have been appropriate for the task at hand (the source code and set of available loss functions within Keras can be seen [here](https://github.com/keras-team/keras/blob/c2e36f369b411ad1d0a40ac096fe35f73b9dffd3/keras/metrics.py)) Similarly, we investigated using different metrics for model performance during training, as 'binary accuracy' alone (aliased to 'acc' in the Keras optimizer implementations, as seen in figure 4) may not represent a comprehensive description of the models performance. This too however, much like the loss function, proved to be one of the few available that was well suited to the task, as other metrics that we tried (full list and source code available [here](https://github.com/keras-team/keras/blob/master/keras/metrics.py)) yielded either uninformative and loosely correlated disgnostics (e.g. cosine proximity, hinge) or did not provide any new or more insightful information than binary accuracy (e.g. mean absolute error, mean squared logarithmic error). For the future, we will investigate implementing custom metrics or using metric libraries from non-Keras sources ([this](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics) appears to be a promising jumping off point)

---
---
+ ### <a name="selected-models"></a> 3. Evaluation of Selected Models
One of the final steps within our model generation software was the option to preserve a model and its weights, labelled with the hyperparameters and conditions it was created within, if it performed adequately during the model evaluation. We have preserved a handful of models, due to either quality of performance or significance in progress, and these will be briefly discussed here, with information regarding how they were created, how they performed on our data, and additional notes, if necessary. Because of our comparatively broad use of number of epochs as a model benchmark (see discussion on [epochs](#nr-epochs)) and because the number of epochs directly corresponds to the amount of learning and processing time taken to generate the model, the models will be referenced here (as they are in their repository) by this number and the data set they were trained over.

---
---
#### 1. 96 epoch model
* **Trained over:** The full, unbalanced drone set, augmented
* **Learning Rate:** 7*10^-4
* **Batch Size:** 32
* **Performance over 'boom' set:** 79.14%
* **Performance over 'drone' set:** 73.70%
* **Performance over 'handheld' set:** 57.41%
* **Notes:** This is the first model trained without human supervision and trained over a period longer than 16 epochs. It represents our transition from the model debugging phase of this project to the model analysis phase
#### 3. 128 epoch model
* **Trained over:** The balanced boom set, augmented
* **Learning Rate:** 7*10^-4
* **Batch Size:** 32
* **Performance over 'boom' set:** 59.17%
* **Performance over 'drone' set:** 61.91%
* **Performance over 'handheld' set:** 56.07%
* **Notes:** This set represents our first exploration of data distribution manipulation and cleaning. Within our framework, a set can be 'balanced' by evening out the number of healthy and sick examples in the set, reducing the set by the amount of excess of one, in comparison to the other. Despite this being claimed to be a technique for eliminating overfitting, it did not seem to have such an effect in our trials, and appeared to negatively impact the model performance proportionately to how many images were removed from it, indicating that quantity of data matters much more for model performance than quality does. Curiously, in this model, as with many other models we trained over the boom set, it performs better over the drone data than over the data it was trained over; this may indicate that the drone data is in some way better put together. 
#### 3. 800 epoch model
* **Trained over:** the full, unbalanced boom set, with augmentation
* **Learning Rate:** 7*10^-4
* **Batch Size:** 32
* **Performance over 'boom' set:** 64.32% 
* **Performance over 'drone' set:** 69.29%
* **Performance over 'handheld' set:** 56.24% 
* **Notes:** This is the first 'large model' we created, meant to observe longer term changes in model training and the benefits of longer training times, model performs better than smaller counterparts, but this does not scale with time (diminishing returns), this is a trend seem in all large models.
#### 4. 2048 epoch model
* **Trained over:** the full, unbalanced drone set, with augmentation
* **Learning Rate:** 1*10^-4
* **Batch Size:** 32
* **Performance over 'boom' set:** 78.88%
* **Performance over 'drone' set:** 93.44% accurate
* **Performance over 'handheld' set:** 56.63% accurate
* **Notes:** This is the biggest model we have trained to date and is the one we are using at the present within our downstream processing of live crop data. It performs with greater accuracy than expected over the boom set, despite being trained only over the drone set; this is a favorable result for the quality of this model as well as the generalizability of our data and results in general 
---
---
