## Cellular Neural Network (CNN) Boundaries

After doing the CNN simulation problems that can be seen in the CNN_simulation directory we decided to take the simulations one step further. The simulation only thresholded the inner pixels and we wanted it to threshold the outer pixels as well. In order to threshold the outer pixels we needed to implement boundary conditions. We did this in three different ways:

1. **Fixed:** both input value and state have predefined fixed values between -1 and 1
2. **Zero-flux:** instead of nonexistent cells use the value of the nearest existing cell
3. **Toroid:** use the idea of a [torus]
	* If you go out of the image on the top then the cell's value from the last row should be used, etc.

## Use
* Simply execute the scripts
* When done viewing the image click on the output image box and push any key to exit the program

## Other Notes
* These codes only differ in the CellEquation part of each code
* The other two files in this directory are the input images used for the boundary simulations

[torus]: https://en.wikipedia.org/wiki/Torus
