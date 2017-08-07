# Upper ReLU

This was a project that I did not spend too much time on but would stay informed about as my research partner spent a lot of time on it. John came up with the idea of adding a nonlinearity that not only capped the data points at -1 but also at 1. This idea came from the nonlinearity function that is used in CNNs, which is similar to the [sigmoid curve]. The changes that needed to be made to the networks in order to test this idea were actually very simple and mostly involved adding another ReLU to the convolutional network. In this directory you can see a bunch of networks that have the upper ReLU idea implemented. After a lot of testing John found that by adding the upper ReLU the network learned much faster without much change in eventual accuracy, which were very promising results. To get more information on this project refer to [John's GitHub page on the topic].

[John's GitHub page on the topic]: https://github.com/jmcguinness11/CNN_research/tree/master/UpperRelu

[sigmoid curve]: https://en.wikipedia.org/wiki/Sigmoid_function
