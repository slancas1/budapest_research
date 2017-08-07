# Cellular Neural Network (CNN) Simulations

In order to better understand how CNNs work we worked on these two different CNN implementations.

*  **thresholding_CNN.py:** This was a simple implementation of an already existing template which thresholds the colors of pixels to either white or black. An explanation of the template can be found in [this paper].
* **maze.py:** During this simulation instead of globally solving a maze we showed that the problem can be solved locally by creating an algorithm. There were two parts to solving the maze:
	1. Decrease the labyrinth at each step by removing the end points --> in order to do this we had to create our own CNN template
	2. Put the two pixels represented in the inital state input image back to the output image

## Use
* Simply execute the scripts
* Two image windows will appear, one displaying the input image and one displaying the output image
* When done viewing the images click on one of the image boxes and push any key to exit the program

## Other Notes
* All of the .bmp files in this directory are the input images for the simulations
	* **maze.py:** Init.bmp and labyrinth.bmp
	* **thresholding_CNN.py:** avergra2.bmp or concave.bmp

[this paper]: http://cnn-technology.itk.ppke.hu/Template_library_v3.1.pdf   
