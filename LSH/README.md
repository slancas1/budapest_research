# Locality Sensitive Hashing (LSH)

For the last few weeks of the summer I worked on implementing and testing [Locality Sensitive Hashing (LSH)] in some of our already existing networks. This was a very long and extensive process. Once we got a working implementation in one of our one-shot learning networks I decided to test the benefits of the LSH by running timing and accuracy tests. In order to make the testing as comprehensive as possible I implemented the LSH metric in one-shot code for three different data sets (MNIST, CIFAR-10, and Omniglot).

## About LSH

Locality-sensitive hashing is a metric that works to reduce the dimensionality of high-dimensional data which is something that is potentially beneficial in neural networks which require large amounts of high-dimensional data. LSH works by hashing input data to buckets and does so in a way that increases the likelihood of collisions between similar items. By doing this it becomes easier to find the nearest neighbor to a query element because only the elements in the same bucket need to be considered. For more background information about LSH refer to the [Wikipedia article] or to the [papers] portion of this repository for a useful papers on the topic. 

## LSH Implementation

I worked closely with my supervisor, Andras Horvath, to create a working implementation of the LSH metric. We decided that it made the most sense to implement LSH in a one-shot learning network because the idea of using select supports and a single query element made the most sense with the hashing. In the LSH implementation the supports and the query are first projected onto a series of planes as described in this [paper]. After this initial projection comparisons are done to create the hash of the element which is simply a vector of zeros and ones. The hash is created based on the element's location relative to the different planes. After the elements are hashed the distance between the two elements is calculated. This distance calculation can be done using a number of different distance metrics which are all listed below. SOMETHING ABOUT LOSS. Finally, an additional optimizer was added to train the planes that are used to hash the elements. 

## Results

After running extensive tests on the LSH codes and other one-shot networks which utilize cosine similarity I found that the LSH yielded very promising results. All of the results from the different tests that I ran can be found at [this link]. As can be seen in the tables that are linked above, by adding LSH the time needed to classify a query element was significantly decreased, sometimes by as much as 50%. While the accuracy results that were yielded for the LSH one-shot implementation were not as conclusive it appears that the LSH may reduce the accuracies slightly. However, my supervisor and I discussed that there are many different parameters that we could change which may increase the accuracy of the LSH networks, so this is something that should be looked into further. The graphs below show some of the results that I found when comparing LSH and cosine similarity in one-shot networks.

<center><img src="https://github.com/slancas1/budapest_research/blob/master/pictures/numsuppstime.png" width="700" height="427.778" /></center>
<center><img src="https://github.com/slancas1/budapest_research/blob/master/pictures/accandtime.png" width="700" height="449.495" /></center>

## Future Work

Due to the fact that I started this work later on in the summer, I did not have time to run nearly as many tests as I would have liked regarding this subject area. First of all, all of the tests that I ran on these networks were only for two classes which is why a lot of the accuracies that can be seen are relatively high. Differentiating between two classes is a way less complicated problem then classifying multiple classes. On the last few days of my stay here in Budapest I started to run tests on the LSH MNIST code for 10 classes, but was not able to obtain all of the data I would have liked to due to lack of time. One thing I would do if I continued this work would be to change the codes so that they were all compatible for any number of classes and run the networks for a higher number of classes. The number of classes is just one of a number of parameters that exist in these networks that could be changed in order to investigate the affects of LSH. A list of all of the parameters can be seen below: 

### List of Parameters

* **Distance metrics:** The distance metrics used to calculate the distance between the hashed query element and hashed supports is one parameter that could be investigated. I ran some tests where I changed this parameter but the tests were not very conclusive. I would be interested to do some more in-depth research concerning this parameter as I am interested to see if it really has an effect. The different distance metrics are listed below:
	1. Euclidean
	2. Squared Euclidean
	4. Centered Euclidean
	5. Hamming
	6. L1 Norm

* **Number of images used for one-shot learning:** By increasing the number of images that are fed into the network the accuracy should increase because then there are more elements for the query to be compared to. By the same measure the time needed to classify the element should also increase. 

* **Number of supports per class:** Increasing this parameter should increase accuracy and time due to the same reasoning as explained above. 

* **Length of LSH hashed vectors:** By increasing the length of the LSH hashed vectors the hashes that are produced are allowed to be more specific which means the accuracy should increase because there are now more potential combinations of hashes to choose from. The time should also slightly increase because more ones and zeros will need to be compared between the two hashed vectors. 

* **Network complexity:** Like almost all neural networks the structure of the network and its complexity is always a parameter that can be adjusted. Usually by making the networks more complex by increasing the number of kernels in each layer, etc. the network is made more accurate. I would be curious to see if the same applies when incorporating LSH.

The final list I am going to include is a list of tests that I would conduct if I had more time to spend with this research. 

### Potential Future Tests

* Run all tests for more iterations as this might increase accuracy (only used 15,000 iterations for these results due to lack of time)
* Run all of the timing / accuracy tests for 10 classes in all three data sets
* Get accuracies for different number of supports (only have times for this right now)
* Get times for different distance metrics (this is something that would not be very hard to do but I would just be interested to see if there was any real difference)
* Change LSH networks to use longer hash vectors and see how this affects accuracy and time
* Change LSH networks to be more complex and see how this affects accuracy and time
* Increase number of images in training data set and see how this affects accuracy and time

[Locality Sensitive Hashing (LSH)]: https://en.wikipedia.org/wiki/Locality-sensitive_hashing
[timing and accuracy tests]: https://docs.google.com/spreadsheets/d/1e_XAS9H61Q7nniCcwsgHAG-JApPW_IJv_6JAgzxbGMw/edit#gid=0
[Wikipedia article]: https://en.wikipedia.org/wiki/Locality-sensitive_hashing
[paper]: https://github.com/slancas1/budapest_research/blob/master/papers/Similarity%20Search%20and%20Locality%20Sensitive%20Hashing%20using%20Ternary%20Content%20Addressable%20Memories.pdf
[papers]: https://github.com/slancas1/budapest_research/tree/master/papers
[this link]: https://docs.google.com/spreadsheets/d/1e_XAS9H61Q7nniCcwsgHAG-JApPW_IJv_6JAgzxbGMw/edit
