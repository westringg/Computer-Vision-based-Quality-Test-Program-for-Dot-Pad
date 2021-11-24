# Computer-Vision-based-Quality-Test-Program-for-Dot-Pad

## Project Description
@verson 2021.07.31
This program tests Dot Pad, a tactile device for the blind developed by Dot Incorporation. Dot Pad is composed of 300 Cells and each cell contains 8 Pins. Each pin can either come up or go down, depending on the hex data sent through serial port. Thus, Dot Pad can display several patterns distinguishable by touching and the aim of the program is to test rather the pattern is properly displayed, based on computer vision technology.

## Code Structure
![Code Structure](https://user-images.githubusercontent.com/68358806/142944256-3b265a89-ab53-4aac-b3c6-bed22b9acf1a.png)

## Files Descriptions

-Alternative Approaches: houghCricle_BGR_updated.py can be replaced by the following files
  - gridCalib.py : An alternative approach to manually get a fixed grid that fits every pin of Dot Pad, rather than automatically detecting it.

  - houghCircles_BGR_updated.py : An alternative appr

houghCircles_HSV_8pic.py
Update houghCircles_HSV_8pic.py
17 minutes ago
houghCircles_resize.py
Update houghCircles_resize.py
15 minutes ago
main.py
Update main.py
15 minutes ago
pattern_control.py
