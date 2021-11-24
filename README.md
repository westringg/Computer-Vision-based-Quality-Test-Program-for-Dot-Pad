# Computer-Vision-based-Quality-Test-Program-for-Dot-Pad

## Project Description
@verson 2021.07.31
This program tests Dot Pad, a tactile device for the blind developed by Dot Incorporation. Dot Pad is composed of 300 Cells and each cell contains 8 Pins. Each pin can either come up or go down, depending on the hex data sent through serial port. Thus, Dot Pad can display several patterns distinguishable by touching and the aim of the program is to test rather the pattern is properly displayed, based on computer vision technology.

## Code Structure
![Code Structure](https://user-images.githubusercontent.com/68358806/142944256-3b265a89-ab53-4aac-b3c6-bed22b9acf1a.png)

## Additional Files
-Alternative Approaches
  - gridCalib: An alternative approach to manually get a fixed grid that fits every pin of Dot Pad, rather than automatically detecting it.

  (houghCricle_BGR_updated.py can be replaced by the following files)
  - houghCircles_HSV_8pic: Uses HSV instead of BGR as the colour system.
  
  - houghCircles_bySize: An alternative approach to distinguish each pin status(up/down) by its size appearing in the picture rather than its colour or brightness.
  
  - houghCircles_resize: An initial trial to improve the accuracy of detecting circles.
