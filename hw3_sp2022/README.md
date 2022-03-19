We started with the design from 1.6, since convolutional networks are better suited for processing images than linear ones.
We removed the last conv layer and replaced it with linear layers from 1.5. We thought it would imporve the accuracy, since the accuracy we got from 1.5 is higher than that we got from 1.6.
To further imrpove the accuracy, we added another sequence of conv layers.
For simplicity, we let the model to accpet input of all the parameters of the layers in this network, so that we can modify them more easily for Hyperparameter searching.
We experimented with various sets of hyperparameters for the network and found that it performs best when we have something like:
![Presentation1](https://user-images.githubusercontent.com/77927150/159139197-04167b22-78cf-48b5-8ec6-4cce91c0e0f9.png)
Also, we found that when the hidden layer width in the linear sequence is set to 100, we have higher accuracy.
We then added dropout after the two poolinf layers, but it did not improve accuracy. So we did not add dropout.
We found the adding a ReLU layer between the two linear layers may lead to overfitting, so we left that out as well.
We then tried different LR, bs, and epochs. We found a balance point when LR = 0.01, bs = 400, and epochs = 2000. This gives us an accuracy of 92. The run time is roughly 420s.
