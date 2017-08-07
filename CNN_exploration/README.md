# Cellular Neural Network (CNN) Exploration

We worked with CNNs towards the end of our training. We did the introductory exercises that can be found in this directory in order to gain a basic understanding of CNNs. Our supervisor wanted us to have this foundation because later in the summer we worked on creating a CNN friendly Convolutional Neural Network (CoNN) which can be found [here].

## Basic CNN Background

The main difference between Cellular Neural Networks (CNNs) and normal Convolutional Neural Networks (CoNNs) is that all of the operations that occur in CNNs are local operations, meaning that communication is only allowed between neighboring units. This quality of CNNs makes their implementations extremely efficient. An important quality of CNNs is that they are structured as a MxN grid of units which is why they can only communicate with direct neighbors (SEE PICTURE BELOW). Feel free to browse the [Wikipedia page on CNNs] to find out more information.



<center>
<img src="https://github.com/slancas1/budapest_research/blob/master/pictures/cnns.png" width="400" height="377.41" />
</center>


[here]: https://github.com/slancas1/budapest_research/tree/master/CNN_friendly_CoNN
[Wikipedia page on CNNs]: https://en.wikipedia.org/wiki/Cellular_neural_network
