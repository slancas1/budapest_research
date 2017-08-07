# One-shot Learning Networks

A major problem with machine learning is that most networks need to be presented with very large amounts of data to learn effectively. Because of this, another project we worked on this summer was implementing [one-shot learning] networks. One-shot learning attempts to solve the problem mentioned above by trying to learn from much smaller sets of data which is often more realistic. Our networks are based off [this paper] from Google DeepMind. I wrote three one-shot networks for each data set that I was working with this summer (MNIST, CIFAR-10, and Omniglot). The Omniglot data set is typically the data set that is used in the one-shot literature so it was not too hard to create a network that worked for this data set. However, when it came to creating the one-shot networks for MNIST and CIFAR I had to write a program that only selected a few images for the training data sets because if the full data sets were used for training then the networks would not be one-shot. The toy problem network uses a validation set and goes back and chooses the best supports to determine the best support accuracy.

## Network

The one-shot network that we created includes three convolutional layers and does not utilize a fully-connected layer. This means that the network outputs feature maps. In our network we first select a query class and run a single element of this selected class through the network. After this we pick two supports from each class (including the query class) and run them through the same network. Once all of the necessary elements have been run through the network we calculate the [cosine similarity] between the query and all of the supports. We then average the similarities found between the query and the supports in the same class. Finally, the class of the query is guessed by using the class of the group of supports that were most similar to the query element. 

## Results

The results that I found using these networks can be seen at the following [link]. I have been comparing the results of these networks to the results of my [LSH networks]. The accuracies achieved using one-shot learning are usually very hard to generalize. Furthermore, one-shot networks tend to be overfit which means that they can achieve very high training accuracies but much lower testing accuracies which can be seen in the tables linked above. 

## Future Work 

The future work that I plan to do with the LSH research involves running simultaneous tests with these networks. More details about the LSH futue work can be found [here].  

[cosine similarity]: https://en.wikipedia.org/wiki/Cosine_similarity
[one-shot learning]: https://en.wikipedia.org/wiki/One-shot_learning
[LSH one-shot accuracies]: https://github.com/slancas1/budapest_research/tree/master/LSH
[this paper]: https://github.com/slancas1/budapest_research/blob/master/papers/OneShotLearning1.pdf
[link]: https://docs.google.com/a/nd.edu/spreadsheets/d/1e_XAS9H61Q7nniCcwsgHAG-JApPW_IJv_6JAgzxbGMw/edit?usp=drive_web
[LSH networks]: https://github.com/slancas1/budapest_research/tree/master/LSH
[here]: https://github.com/slancas1/budapest_research/tree/master/LSH
