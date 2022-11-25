# Digit-Recognizer - Keras

### Download dataset
You can find the dataset use to train the AI [here](https://www.kaggle.com/competitions/digit-recognizer/overview)  

### Quick explanation
There is 2 class for this project

The first one is the menu, that manage the windows using tkinter.  
And the second one is the AI, that manage the Keras model, the training and the prediction.

To use the application, run main.py

### The app
The application look like that:

### To train an AI
Use both button "Open the train file" and "Open the test file" to load the dataset dw on kaggle.  
Choose the number of epoch and the batch size (recommended to only change the number of epoch, 5 epoch is enough to get a good AI).  
Then click on "Start training AI".  

After training, you can use or save the AI.

### To use an AI
To use an AI you need to train one or load one using the button "Load model".  
Then you can draw the number using the mouse in the white cube under the "clean" button.  
The result is under the canvas and you can clean the canvas using the "clean" button.
