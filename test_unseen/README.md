# Testing for Non-Trained Classes

One question that I became intersted in this summer was what would happen if the network was tested for classes of data that it was not trained for. I was curious to see what would happen in a network with a FCL and a network without a FCL. I decided to test this idea using the MNIST data set. In order to research this idea I created code that would train the network and save the weights and then another code that would read in those weights and test the network with unseen classes. What we were hoping to see was that the non-trained classes in the network without the FCL would end up more in the middle instead of tending to one class like they do in a network with a FCL. This is what was eventually seen as is demonstrated by the graphs included below. The two subdirectories contain the training and testing codes for both the networks with and without the FCL. Each subdirectory also contains a create_graph.py script which creates a visual representation of the results that were found. This was a project that I really enjoyed working on this summer.

## Creating the Graphs

In each case in order to create the graphs I saved a list of values into a csv file from the code I used to test the networks. In the network with the FCL I saved the label of the data that was being classified as well as the predicted weights in favor of each trained class. In the network without the FCL I saved the label of the data that was being classified and the weights of that data when compared to maps of negatives ones and ones. By saving this information into a csv file I was able to write a script that read in the data and then created the graphs accordingly. 

## Results

As mentioned above the results that we expected to see from this research were mostly seen. The fact that the network without the FCL tends to classify the unseen classes more in the middle is another potential benefit to eliminating the FCL from networks. When preparing to construct both of the graphs that can be seen below the networks were trained for classes 6 and 9 and then tested for all of the remaining MNIST classes. As seen in the graphs the other classes in the graph for the network without the FCL tend more towards the middle (reference line) while the other classes in the graph for the network with the FCL tend to be on either side of the reference line. Of course there are a few exceptions to this general trend, but we were happy to see that the trend mostly held. 

<center><img src="https://github.com/slancas1/budapest_research/blob/master/pictures/noFCLunseen.png" width="500" height="373.03" /></center>

<center><img src="https://github.com/slancas1/budapest_research/blob/master/pictures/FCLunseen.png" width="500" height="371.97" /></center>

