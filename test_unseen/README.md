# Testing for Non-Trained Classes

One question that I became intersted in this summer was what would happen if the network was tested for classes of data that it was not trained for. I was curious to see what would happen in a network with a FCL and a network without a FCL. I decided to test this idea using the MNIST data set. In order to research this idea I created code that would train the network and save the weights and then another code that would read in those weights and test the network with unseen classes. What we were hoping to see was that the untrained classes in the network without the FCL would end up more in the middle instead of tending to one class like they do in a network with a FCL. This is what was eventually seen. The two subdirectories contain the training and testing codes for both the networks with and without the FCL. Each subdirectory also contains a create_graph.py script which creates a visual representation of the results that were found. This was a project that I really liked working on this summer.

## Use 

TO DO

## Results

TO DO -- add different graphs
