# CS591LFDTermProject
Spring 2019, CS591 Learning From Data Term Project

This is a convolutional neural network using keras, on top of tensorflow, to classify images.

The dataset can be downloaded [here](https://drive.google.com/open?id=1bXZzYlsX0USmF4_Wm8QcxgNJ8RJkiPil). After you put the parent folder somewhere, you need to create system variables for the training set and testing set. The training set variable is named 'CS591LFDTRAINDIR'. The testing set variable is named 'CS591LFDTESTDIR'. The variables are absolute directories of the datasets.


Future plans:
- Using a linear algorithm to increase the number of nodes in the hidden layer. Not sure of the stopping point, as of right now, except for a definite number of iterations.
- Using a genetic algorithm to design the CNN itself, between iterations. The number of hidden layers, and number of nodes per hidden layer, will be pseudo-randomly determined, based on the algorithm. The activation function for the nodes will always be the same.
- Each algorithm will be in its own branch for better organization. 
- The overall dataset will be changed.
   - The smaller subset will become 4k images.
   - The subsets will be joined into 1.
      - A definite amount will be pulled out, randomly, before the training stage, to be used for the validation stage. 
      - Around 4k images again?
      - This will help further remove unintended bias in the dataset, besides the shuffling.
