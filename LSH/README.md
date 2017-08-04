## Locality Sensitive Hashing (LSH)

For the last few weeks of the summer I worked on implementing and testing [Locality Sensitive Hashing (LSH)] in some of our already existing networks. This was a very long and extensive process. Once we got a working implementation in one of our oneshot learning networks I decided to test the benefits of the LSH by running [timing and accuracy tests]. In order to make the testing as comprehensive as possible I ran the LSH oneshot code using three different distance metrics to calculate the distance from the query to the closest support and did this for three different data sets. 

ADD WAY MORE BACKGROUND INFO ABOUT LSH

INCLUDE LSIT OF PARAMETERS

### Distance Metrics
1. Euclidean
2. Squared Euclidean
3. L1 Norm

### Data Sets
1. MNIST
2. Omniglot
3. CIFAR-10

As can be seen in the table linked above, by adding LSH the time needed to classify a query element was significantly decreased. However, the accuracy results that were yielded for the LSH oneshot implementation were pretty inconclusive which is not unusual for oneshot networks. In order to more concretely test the affects of LSH on accuracy I started to implement the LSH metric in on of our other networks, specifically a network without a Fully Connected Layer (FCL). I did not get to finish this part of the research before I had to leave but may continue with it as the results would be very interesting.

## Network

TO DO

## Data Sets

TO DO

## Use

TO DO

## Results

TO DO -- link the google sheets and include graphs

## Future Work

TO DO

[Locality Sensitive Hashing (LSH)]: https://en.wikipedia.org/wiki/Locality-sensitive_hashing
[timing and accuracy tests]: https://docs.google.com/spreadsheets/d/1e_XAS9H61Q7nniCcwsgHAG-JApPW_IJv_6JAgzxbGMw/edit#gid=0
