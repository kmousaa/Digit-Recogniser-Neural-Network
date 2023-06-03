# Digit-Recogniser-Neural-Network
[![Generic badge](https://img.shields.io/badge/language-python-orange.svg)](https://shields.io/)
[![GitLab last commit](https://img.shields.io/github/last-commit/kmousaa/Digit-Recogniser-Neural-Network)](https://img.shields.io/github/last-commit/kmousaa/Digit-Recogniser-Neural-Network)
[![Generic badge](https://img.shields.io/badge/completion-complete-blue.svg)](https://shields.io/)


This project uses a neural network in order to recognise handwritten digits (from 0 - 9)

# Dataset
The MNIST dataset is used for training and testing the model. It consists of a large number of handwritten digits from 0 to 9, along with their corresponding labels. The dataset is automatically downloaded and split into training and test data when running the code.

# Installation
1. Clone the repository: `git clone https://github.com/kmousaa/Digit-Recogniser-Neural-Network.git`
2. Install the required dependencies: `pip install tensorflow opencv-python numpy matplotlib`

# Running
1. Run the python script `main.py`
2. A window will be displayed showing the handwritten digit from the "handwriten_numbers" folder. 
3. The neural network's prediction for the digit will be printed in the terminal. 

# Adding your own digits
1. Create a PNG file of size 28x28 pixels representing the handwritten digit you want to add. You can do this by scanning a paper or using microsoft paint, and then using an online tool such as [28x28 resizer](https://www.imageresizeonline.com/convert-image-to-28x28-pixels.php) to make sure the image is the correct size.  Ensure that the image format is PNG and follow the guidelines mentioned below.
2. Save the PNG file with the filename format "digit[num].png", where [num] is the next number after the existing digits. For example, if the last digit file is "digit9.png", name your file "digit10.png" for the next digit.
3. Copy the PNG file into the "handwriten_numbers" folder in the project directory.
4. Run the `main.py` script

# Changing the model
The model has already been trained, and is now commented out to prevent re training it every time the code is run. 
To modify the model architecture, follow these steps:

1. Open the `main.py` file
2. Uncomment the code section that creates the model.
3. Customize the model architecture by adding or removing layers and adjusting the number of neurons, or the number of times to train the model (documented by the comments)
4. Run the `main.py` script to train the model with the new architecture.
5. Comment out the code in order to not re-train the model every time 
6. You can now use the new model!

# Problems
The current neural network model for handwritten digit recognition faces the following challenges:

1. Accuracy: The model is not very accurate (70% accurate), leading to occasional mistakes.
2. Overfitting: The model tends to overfit the training data, resulting in poor performance on new, unseen images.
3. Confusion between 1 and 7: The model struggles to differentiate between the digits 1 and 7, which have similar shapes.
4. Optimal architecture: Determining the ideal number of layers and neurons for the model is difficult.

I will try on improving the model to improve its accurary. Feel free to expriment and change the model and let me know if you were able to improve it!
