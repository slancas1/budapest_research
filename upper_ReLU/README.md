## Upper ReLU

This was a project that I did not spend too much time on but would stay informed on as my research partner spent a lot of time on it. John came up with the idea of adding a nonlinearity that not only capped the data points at -1 but also at 1. This idea came from the nonlinearity function that is used in CNNs, which is similar to the [sigmoid curve]. The changes that needed to be made to the networks in order to test this idea were actually very small and mostly involved adding another ReLU line to the convolutional network. In this directory you can see a bunch of networks that have the upper ReLU idea implemented. After a lot of testing John found that by adding the upper ReLU the network learned much faster without much change in eventual accuracy, which were very promising results.

JUST LINK TO JOHN'S PAGE ON THIS

[sigmoid curve]: https://en.wikipedia.org/wiki/Sigmoid_function
