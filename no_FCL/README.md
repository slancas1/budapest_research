# Networks Without a Fully-Connected Layer (FCL)

One of the first tasks we worked on this summer was removing the FCL from our CoNNs. We did this because the FCL is not CNN friendly and is very computationally expensive. Using an FCL means global calculations are done because each element is compared to every other element and this is not CNN friendly because CNNs use local operations. The two subdirectories in this directory include networks without a FCL for both [2 MNIST classes] and [10 MNIST classes]. Within these directories the different files only differ by how the loss is calculated.

## About the FCL

In a normal CoNN the convolutions output a certain number of feature maps, and after the last convolutional layer a fully-connected layer exists to "connect" every neuron from the previous layer to the next layer. As explained above this operation is very computationally expensive. 

![FCL](https://github.com/slancas1/budapest_research/blob/master/pictures/fcl.jpeg)

## Removing the FCL

In order to remove the FCL from our networks we fixed the number of output feature maps from the final layer of the network to the number of classes. After this we created maps of all negative ones or all ones which were the same size as the maps that were outputted from the network. We then did something similar to one-hot encoding which would make the map corresponding to the label all ones and all other maps would be negative ones. The accuracy and loss calculations could then be performed using these maps. The networks without a FCL performed the operations much faster, but there was definitely a drop in accuracy that could be seen.

[2 MNIST classes]: https://github.com/slancas1/budapest_research/tree/master/no_FCL/no_FCL_2

[10 MNIST classes]: https://github.com/slancas1/budapest_research/tree/master/no_FCL/no_FCL_10
