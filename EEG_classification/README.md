## EEG Classification

We decided that in order to apply all that we learned this summer we would work on a side prproject using the resources the university could offer us. One of the labs at the university had access to [EEGs], which is a device that can test electrical activity in the brain. Once given access to this device we worked on recording and parsing data so that we could decide what type of network to try the problem on. We decided that because it was not reasonable to collect a ton of data using the EEG, the problem would work best in a oneshot learning network. We eventually tried to classify three different problems using the EEG. 

1. Collected data of four different people focusing on the same image and tried to distinguish between people.
2. Collected data of one person focusing on a slideshow that alternated between pictures of storms and pictures of beaches and tried to distinguish between the two classes.
3. Collected data of one person relaxing and then doing math questions and tried to distinguish between the two classes. 

While we were able to get some promising results when it came to the first problem, classifying people, we found that the network was not really able to classify better then chance for the other two problems. We think this could have been because the data that was collected was not different enough between the two classes in each given case. Regardless, this was a very cool application to spend some time on this summer.

## Files in Directory

* **Four Problem Solving Network Files:** classify_math_relax.py, classify_person_2.py, classify_person_4.py, classify_weather.py
* **Two Files for Loading Data:** load_5s_data.py, load_data.py
* **Three Subdirectories Containing Data:** people_data, relax_math_data, weather_data 

[EEGs]: https://en.wikipedia.org/wiki/Electroencephalography 
