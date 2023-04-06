# Keyboard Inspection Project Using OpenCV

This is a Python script that detects the dimensions of objects in a live video feed and also finds missing keys on a keyboard. The script uses OpenCV, an open-source computer vision library, and SIFT algorithm for object detection.

### Requirements

- Python 3.6 or above
- OpenCV library
- Numpy library

### Running the script

To run the script, open your terminal, navigate to the directory where the script is located, and run the following command:

```
python detection.py
```

The script will launch and start capturing video from your computer's webcam. It will detect the dimensions of the keyboard and draw a box around it. It will also look for missing keys on the keyboard and draw a square around them.

### Script Details

The script has the following functions:

- `open_camera()` - opens the webcam for video capture and returns the video feed.
- `find_contours(frame)` - finds contours in the video feed and returns the contours in the frame.
- `find_dimension(frame)` - detects the dimensions of the keyboard and draws a box around it.
- `find_missing_keys(frame, gray_frame, keyboard_type)` - looks for missing keys in the keyboard and draws a square around them.

The script first loads the SIFT algorithm and the aruco detector dictionary and parameter. It then opens the webcam for video capture and resizes the output window.

The `find_contours(frame)` function converts the video feed to grayscale and creates an adaptive threshold mask for the video feed. It then finds contours in the video feed and filters smaller objects less than 2000 from contours.

The `find_dimension(frame)` function detects the aruco marker and draws a square box around the aruco marker. It then finds the perimeter of the aruco marker, which is used to find the pixel to cm ratio of the aruco. It then gets contours from the video feed and draws objects boundaries with minAreaRect function. It calculates the object width and height in pixel to cm ratio and converts it to mm. If the object's height or width is greater than 42mm, it marks the center of the object and draws a rectangular box around it. It then displays the object's width, height, and angle of rotation.

The `find_missing_keys(frame, gray_frame, keyboard_type)` function takes the frame, gray_frame, and keyboard_type as input parameters. It loads the template image based on the keyboard type and converts it to grayscale. It then finds the width and height of the template image and performs template matching. It uses a threshold value to filter out weak matches and draws a rectangle around the object in the frame.

### Output

See sample output here 👇:

![Output](demo.mp4)
