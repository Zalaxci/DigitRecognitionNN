## 1. Hyperparameter Tuning: Try adjusting number of hidden layers, hidden units, activation function, learning rate, number of epochs, etc.

I tried tweaking the hidden layers (number of layers & number of neurons). After the reference (best performing model), with the activation function constant the best performing models were the one with 1 layer of 512 neurons and the one with 2 layers of decreasing number of neurons (384 and 256).

### Accuracy per model using default (tanh) activation function
| Structure | 5th Epoch | 10th Epoch | 15th Epoch |
| --- | --- | --- | --- |
| ***1x512 neurons*** | 0.9305 | 0.9512 | 0.9596 |
| ***2x256 neurons (reference)*** | 0.9518 | 0.9568 | 0.9628 |
| ***3x256 neurons*** | 0.9159 | 0.9367 | 0.9498 |
| ***2x128 neurons*** | 0.8983 | 0.9230 | 0.9353 |
| ***3x128 neurons*** | 0.9009 | 0.9237 | 0.9360 |
| ***384 + 256 neurons*** | 0.9199 | 0.9423 | 0.9542 |
| ***256 + 384 neurons*** | 0.9143 | 0.9352 | 0.9501 |

I proceeded to use the first and third best layer configurations to create 8 new models with 4 different activation functions - the ReLU, ReLU6, GeLU and the Sigmoid. I selected the models with 2 layers since the single layer model seems to have much higher validation loss with most activation functions. Out of the models tested, those with the Sigmoid performed the worse by a huge difference, while the best performers were the ReLU & the GELU.
![image](https://github.com/user-attachments/assets/c207eff6-a1a3-4ec9-8d06-37eb88948d18)
I selected the models 2 of 384 and 256 neurons (better validation accuracy and loss compared to reference structure of 2x256 neuron layers) and ReLU and GELU as activation functions for the following tests.
The final hyperparameter I tried changing is the learning step. With a smaller learning step of 0.0005 I got similar results as with a learning step of 0.001, but with a larger learning step training completely failed and accuracy was approximately as good as guessing.
This is very most likely due to the neural network overshooting towards the direction of steepest descent when and as a result not actually descending down the multidimensional function graph when gradient descent is performed. In other words, whenever a valley with a minimum is found, instead of changing the weights appropriately to navigate towards the bottom of the valley, the training algorithm overshoots, rendering the gradient descent process useless.
What's funny is that the loss is "nan" = "Not A Number" in python, so it seems the cross entropy isn't even calculatable (?) Nonetheless, these are the training results:

| Model | Default learning step (0.001) | LS = 0.0005 | LS = 0.005 | LS = 0.01 |
| --- | --- | --- | --- | --- |
| ***ReLU Model*** | 0.9779 | 0.9819 | 0.0989 | 0.1099 |
| ***GELU Model*** | 0.9791 | 0.9766 | 0.0989 | 0.0989 |

### Finding misclassified digits

We use the afformentioned models to predict every one of the test values using the "x_test" set of matrices as input to the "predict" method, and the results are compared to the true identification ("y_test"). The np.argmax function of numpy is used to find out the digit the AI guess (the one with the largest probability number) as well as the index of the number 1 in the array with the correct classification. A loop is used to print one out of each incorrectly marked digit.

### Quality of the training data - and an attempt at a modification

The MNIST data set is a modified version of the dataset (i.e. set of handwriten digits) of NIST (National Institute of Science and Technology in the USA). The latter consists of numbers written by post office employees as training data, and numbers written by high school students as test data. Thus, it is often considered low-quality or inaccurate data.
Still, since I couldn't find a similar digit databank, I thought of modifying the training and test data (matrices corresponding to pixels in each image). I normalized *x_train* and *x_test*, initially between 0 and 1 (dividing by 255) and then between -0.5 and 0.5 (subtracting 0.5), only to get worse results, unfortunately.
***Accuracy after 5 epochs***
| Model | Default (0-255) pixel values | Pixel Values between 0 & 1 | Between -0.5 & 0.5 |
| --- | --- | --- | --- |
| ***ReLU Model*** | 0.9779 | 0.8885 | 0.8872 |
| ***GELU Model*** | 0.9791 | 0.8795 | 0.8859 |

### Different model structure

Finally I tried reimplementing the digit prediction model using a Convolutional Neural Network structure instead. Convolutional neural networks are known to be best in (at least more complex) image recognition tasks, as a convolution allows one to pick out specific patterns in an image in a more generic way, unrelated to the position (e.g. of a digit) and with less parameters.
As expected, the resulting accuracy is higher than all previous models (0.9887) and so is validation accuracy (0.9769), but even more surprisingly is 0.0735, i.e. much smaller than the 0.2 the afformentioned best perfoming models seem to converge to.
