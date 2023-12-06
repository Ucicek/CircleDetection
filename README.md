# CircleDetection

The objective of this challenge is to develop a circle detector capable of locating a circle in an image with arbitrary noise. The aim is to create a deep learning model that takes an image as input and outputs the location (x,y) of the circle's center and its radius.

# Introduction
To train this model, a basic convolutional neural network (CNN) was utilized, trained on 200,000 data points, each of size 100x100x1. I considered using pretrained model weights, as they could significantly aid in edge detection and circle localization. 
However, as explained later in the 'Challenges' section, this apprach was infeasible for my situation. The model weights are under the folder, pretrained.



# Challenges 
There were variety of challenges faced in this project, these include problems that are related to data, run-time and overfitting the data. 

Since I was working on my local computer to create a well-structured project, I faced with a lot of issues in regards to hardware limiations. The computer I currently have doesn't have a GPU
and it only has avaliable storage of 6 GB. This only permitted me to create a dataset of 30_000 images of size [1,100,100]. I considered using pretrained models, but this would have required sacrificing more storage for images of size [3,224,224] 
and would have significantly increased training time due to the lengthy backpropagation calculations, making it unfeasible.  In order to
solve these sets of problems, I opted in for a smaller model. However, the results I was getting was not the best since I had limited amount of data. 

To resolve this, I modified my data loader to use a generator object, creating data as the images were fed into the model. This allowed me to work with a high amount of data, approximately 
200,000 images, and achieve decent results. At one point, I noticed the loss was not improving, possibly indicating that the model was stuck at a local minimum or a saddle point in higher-dimensional 
space (as true local minima are very rare). To address this, I adjusted the momentum term, which acts like a friction term, helping the model build momentum to overcome saddle points and converge. 
Additionally, I implemented a learning rate reduction make sure the convergence of the model.

# Results

Here are the inital results obtained with 30_000 data with basic CNN. Thresholds of 0.7 and 0.5, respectively.


<img width="219" alt="Screen Shot 2023-12-05 at 4 31 47 PM" src="https://github.com/Ucicek/CircleDetection/assets/77251886/0b479095-e2c0-4a4d-bc6c-852e81900b7e">

<img width="215" alt="Screen Shot 2023-12-05 at 4 31 54 PM" src="https://github.com/Ucicek/CircleDetection/assets/77251886/e9545b4a-fa6e-4f83-a61f-d842cb109d71">

Here are the results achieved by overcoming the challenges described above. Thresholds of 0.5 and 0.7, 0.9, 0.95 respectively.

<img width="212" alt="Screen Shot 2023-12-06 at 1 54 25 PM" src="https://github.com/Ucicek/CircleDetection/assets/77251886/9dfe1492-e33c-48ad-8435-1dad3c95a5ad">
<img width="218" alt="Screen Shot 2023-12-06 at 1 55 35 PM" src="https://github.com/Ucicek/CircleDetection/assets/77251886/3e0a2cf1-f956-4f92-b25b-4082070abebc">
<img width="217" alt="Screen Shot 2023-12-06 at 1 56 49 PM" src="https://github.com/Ucicek/CircleDetection/assets/77251886/f9a469d7-c540-4f22-822c-2d47bd73db9b">
<img width="216" alt="Screen Shot 2023-12-06 at 1 53 07 PM" src="https://github.com/Ucicek/CircleDetection/assets/77251886/b8987343-fe96-496c-a30b-9c9171484cf0">

These are not bad considering only 20 epochs with 10,000 data was used. 


# Getting Started (How to Run)

Copy the repository
```markdown
git clone https://github.com/Ucicek/CircleDetection.git

Download the requirments
```markdown
pip install -r requirements.txt

