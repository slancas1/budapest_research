# EEG Classification

We decided that in order to apply all that we learned this summer we would work on a side project using the resources the university could offer us. One of the labs at the university had access to multiple [EEGs]. An EEG is a device that is used to test electrical activity in the brain. The particular EEG we used was called the MindWave EEG and was produced by the company [NeuroSky]. Once given access to this device we worked on recording and parsing ddifferent types of data to see if we could use one of our existing networks to try and classify the EEG data. In order to read in the data we used an Android app which retrieved the raw values using the EEG and then formatted them into csv format. We decided that because it was not reasonable to collect a ton of data using the EEG, the problem would work best in a one-shot learning network. We eventually tried to classify three different problems using the EEG. 

1. Collected data of four different people focusing on the same imagei for 60 seconds and tried to distinguish between people.
2. Collected data of one person focusing on a slideshow that alternated between pictures of storms and pictures of beaches and tried to distinguish between the two classes.
3. Collected data of one person relaxing and then doing math questions and tried to distinguish between the two classes.

In order to try and classify this data we modified an existing one-shot network that we had to use one-dimensional convolutions because the EEG data was one-deimensional unlike the two-dimensional input images we had been using. 
## Results

While we were able to get some promising results when it came to the first problem, classifying people, we found that the network was not really able to classify better then chance for the other two problems. We think this could have been because the data that was collected was not different enough between the two classes in each given case. Regardless, this was a very cool application to spend some time on this summer.

1. When classifying between people we could achieve about 75% for two people and about 50% for four people which is about 25% better than chance in each case.
2. When classifying between two classes for both the MathRelax and BeachStorm problems we were not able to achieve anything better than chance which was 50%. 

## Files in Directory

* **Four Problem Solving Network Files:** classify_math_relax.py, classify_person_2.py, classify_person_4.py, classify_weather.py
* **Two Files for Loading Data:** load_5s_data.py, load_data.py
* **Three Subdirectories Containing Data:** people_data, relax_math_data, weather_data 

[NeuroSky]: http://neurosky.com/

[EEGs]: https://en.wikipedia.org/wiki/Electroencephalography 
