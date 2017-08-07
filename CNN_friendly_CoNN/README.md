# Cellular Neural Network (CNN) Friendly Convolutional Neural Network (CoNN)

The two sub-directories in this directory include implementations of CNN friendly CoNNs. Some basic background information about CoNNs can be found [here]. In order to make these networks CNN friendly only slight changes needed to be made to the original CoNNs. For example, the convolutional and pooling kernel sizes in a CNN friendly CoNN need to be 3x3. There are implementations of these CNN friendly networks for [2 MNIST classes] and [10 MNIST classes]. This was [work] that the student who came to Budapest the summer before us worked on and so in order to expand on his work we [worked on removing the fully-connected layer (FCL)] from these CeNN-friendly CoNNs because the FCL itself is not very CeNN-friendly.

[work]: https://www.date-conference.com/proceedings-archive/2017/html/7026.html

[worked on removing the fully-connected layer (FCL)]: https://github.com/slancas1/budapest_research/tree/master/no_FCL

[here]: https://github.com/slancas1/budapest_research

[2 MNIST classes]: https://github.com/slancas1/budapest_research/tree/master/CNN_friendly_CoNN/CNN_friendly_2

[10 MNIST classes]: https://github.com/slancas1/budapest_research/tree/master/CNN_friendly_CoNN/CNN_friendly_10
