## Networks Without a Fully Connected Layer (FCL)

One of the first tasks we worked on this summer was removing the FCL from our CoNNs. We did this because the FCL is not CNN friendly and is very computationally expensive. Using an FCL means global calculations are done because each element is compared to every other element and this is not CNN friendly because CNNs use local operations. The two subdirectories in this directory include networks without a FCL for both [2 MNIST classes] and [10 MNIST classes]. Within these directories the different files only differ by how the loss is calculated. 

[2 MNIST classes]: https://docs.google.com/spreadsheets/d/1e_XAS9H61Q7nniCcwsgHAG-JApPW_IJv_6JAgzxbGMw/edit#gid=0

[10 MNIST classes]: https://docs.google.com/spreadsheets/d/1e_XAS9H61Q7nniCcwsgHAG-JApPW_IJv_6JAgzxbGMw/edit#gid=0
