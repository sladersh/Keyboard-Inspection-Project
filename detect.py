import cv2 as cv
import numpy as np

sift = cv.SIFT_create()

# load aruco detector parameter
aruco_parameters = cv.aruco.DetectorParameters_create()

# load aruco detector dictionary
aruco_dict = cv.aruco.Dictionary_get(cv.aruco.DICT_4X4_50)


def open_camera():
    """returns the video feed"""

    # loading the webcam for video capture
    feed = cv.VideoCapture(0, cv.CAP_DSHOW)

    # resizing the output window
    feed.set(cv.CAP_PROP_FRAME_WIDTH, 1350)
    feed.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    return feed


def find_contours(frame):
    """returns the contours in the frame"""

    # convert the video feed to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # create an adaptive threshold mask for the video feed
    mask = cv.adaptiveThreshold(
        gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 19, 5
    )

    # finding contours in the video feed
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # empty contours array
    contours_in_frame = []

    # filter smaller objects less than 2000 from contours
    for contour in contours:
        contour_area = cv.contourArea(contour)
        if contour_area > 2000:
            contours_in_frame.append(contour)

    return contours_in_frame


def find_dimension(frame):
    """finds the dimension of the keyboard and draws box around its border"""

    # detect the aruco marker
    corners, ids, rejected = cv.aruco.detectMarkers(
        frame, aruco_dict, parameters=aruco_parameters
    )

    # check at least one aruco marker is present
    if len(corners) > 0:

        # convert corner values from float to int
        corners_in_int = np.int0(corners)

        # draw square box around the aruco marker
        cv.polylines(frame, corners_in_int, True, (0, 255, 0), 4)

        # find perimeter of the aruco marker
        perimeter_of_aruco_marker = cv.arcLength(corners[0], True)

        # find pixel to cm ratio of aruco (4 x 3.6 = 14.4)
        # width and height of aruco marker used for calibration is 3.6cm
        px_to_cm_ratio = perimeter_of_aruco_marker / 14.4

        # get contours from the video feed
        contours = find_contours(frame)

        # draw objects boundaries with minAreaRect function
        for contour in contours:
            # get the rectangle border
            object_border = cv.minAreaRect(contour)
            (x_center, y_center), (width, height), angle = object_border

            # detect the object borders
            rectangle_box = cv.boxPoints(object_border)
            rectangle_box = np.int0(rectangle_box)

            # get width and height of the object in pixel to cm ratio and convert to mm
            object_width = width / px_to_cm_ratio
            object_height = height / px_to_cm_ratio

            if object_height >= 42 or object_width >= 42:
                # mark the center of the object
                cv.circle(frame, (int(x_center), int(y_center)), 5, (0, 0, 255), -1)

                # draw rectangular box around the object
                cv.polylines(frame, [rectangle_box], True, (255, 0, 0), 2)

                # display the width
                cv.putText(
                    frame,
                    f"Width {round(object_width, 2)}cm",
                    (int(x_center - 80), int(y_center - 15)),
                    cv.FONT_HERSHEY_COMPLEX_SMALL,
                    1,
                    (0, 255, 255),
                    1,
                )

                # display the height
                cv.putText(
                    frame,
                    f"Height {round(object_height, 2)}cm",
                    (int(x_center - 80), int(y_center + 15)),
                    cv.FONT_HERSHEY_COMPLEX_SMALL,
                    1,
                    (0, 255, 255),
                    1,
                )

                # display the angle of rotation
                cv.putText(
                    frame,
                    f"Angle of Rotation {round(angle, 2)}degree",
                    (int(x_center - 80), int(y_center + 45)),
                    cv.FONT_HERSHEY_COMPLEX_SMALL,
                    1,
                    (0, 255, 255),
                    1,
                )


def find_missing_keys(frame, gray_frame, keyboard_type):
    """looks for missing keys in the keyboard and draws a square around it"""

    if keyboard_type == "A":
        missing_template = cv.imread("missing-A.png")
    elif keyboard_type == "B":
        missing_template = cv.imread("missing-B.png")

    # converting template to gray scale
    template_gray = cv.cvtColor(missing_template, cv.COLOR_BGR2GRAY)

    # find the width and height of the template image
    w, h = template_gray.shape[::-1]

    # perform template matching
    res = cv.matchTemplate(gray_frame, template_gray, cv.TM_CCOEFF_NORMED)

    # use a threshold value to filter out weak matches
    threshold = 0.8
    loc = np.where(res >= threshold)

    # draw a rectangle around the object in the frame
    for pt in zip(*loc[::-1]):
        cv.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 1)
        cv.putText(
            frame,
            f"Key missing!",
            (int(pt[0] + w + 10), int(pt[1] + h)),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )


def match_template(gray_frame, frame):
    """performs template matching using sift and brute force matching to match type A, type B keyboards as well as looks for keyboards turned upside down"""

    match_detected = "No keyboard detected"
    min_matches = 9  # minimum number of matches to be considered

    # loading the query images for detection
    query_type_A = cv.imread("type-A.png")
    query_type_B = cv.imread("type-B.png")
    query_upside_A = cv.imread("upside-A.png")
    query_upside_B = cv.imread("upside-B.png")

    # converting query images to gray scale and computing sift key points
    gray_type_A = cv.cvtColor(query_type_A, cv.COLOR_BGR2GRAY)
    query_kp_A, query_des_A = sift.detectAndCompute(gray_type_A, None)

    gray_type_B = cv.cvtColor(query_type_B, cv.COLOR_BGR2GRAY)
    query_kp_B, query_des_B = sift.detectAndCompute(gray_type_B, None)

    gray_upside_A = cv.cvtColor(query_upside_A, cv.COLOR_BGR2GRAY)
    query_kp_upside_A, query_des_upside_A = sift.detectAndCompute(gray_upside_A, None)

    gray_upside_B = cv.cvtColor(query_upside_B, cv.COLOR_BGR2GRAY)
    query_kp_upside_B, query_des_upside_B = sift.detectAndCompute(gray_upside_B, None)

    # computing sift key points of train image (video feed)
    train_kp, train_des = sift.detectAndCompute(gray_frame, None)
    bf = cv.BFMatcher()

    try:
        good_matches_A = []
        good_matches_B = []
        good_matches_upside_A = []
        good_matches_upside_B = []

        # finding template matches
        matches_A = bf.knnMatch(query_des_A, train_des, k=2)
        matches_B = bf.knnMatch(query_des_B, train_des, k=2)
        matches_upside_A = bf.knnMatch(query_des_upside_A, train_des, k=2)
        matches_upside_B = bf.knnMatch(query_des_upside_B, train_des, k=2)

        for m, n in matches_A:
            if m.distance < 0.75 * n.distance:
                good_matches_A.append([m])

        for m, n in matches_B:
            if m.distance < 0.75 * n.distance:
                good_matches_B.append([m])

        for m, n in matches_upside_A:
            if m.distance < 0.75 * n.distance:
                good_matches_upside_A.append([m])

        for m, n in matches_upside_B:
            if m.distance < 0.75 * n.distance:
                good_matches_upside_B.append([m])

        if (len(good_matches_A) > len(good_matches_B)) and (
            len(good_matches_A) >= min_matches
        ):
            match_detected = "Type A keyboard detected"
            find_dimension(frame)
            find_missing_keys(frame, gray_frame, "A")
        elif (len(good_matches_B) > len(good_matches_A)) and (
            len(good_matches_B) >= min_matches
        ):
            match_detected = "Type B keyboard detected"
            find_dimension(frame)
            find_missing_keys(frame, gray_frame, "B")
        elif (len(good_matches_upside_A) >= min_matches) or (
            len(good_matches_upside_B) >= min_matches
        ):
            match_detected = "Keyboard is turned upside down"
        else:
            match_detected = "No keyboard detected"

    except:
        pass

    return match_detected


video = open_camera()

while True:
    # capture video feed frame-by-frame
    ret, frame = video.read()

    # converting the video frame to gray scale
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    match = match_template(gray_frame, frame)

    cv.putText(
        frame,
        match,
        (50, 50),
        cv.FONT_HERSHEY_COMPLEX,
        1,
        (0, 0, 255),
        2,
    )

    cv.imshow("Detect Object", frame)

    # the program will stop when the key 'q' is pressed
    if cv.waitKey(1) == ord("q"):
        break

video.release()
cv.destroyAllWindows()


"""
    README
    
    The flow of the program is explained below:
    
    - First of all it tries to find matching templates from the live video capture feed.
    
    - For matching Type A, Type B and keyboard upside down conditions, SIFT key matching is used.
    
    - We used bfMatcher which is a Brute force technique that tries to compare the templates one by one with the video frame.
    
    - When a match is found for either of the two keyboard types, it looks for the aruco marker in the frame and use it to find the dimensions of the keyboard, and draws the border around it.
    
    - The width, height, angle of rotation and the keyboard type are displayed.
    
    - The program is able to detect the keyboards placed in different orientation as well.
    
    - The program also shows if the keyboard is placed upside down.
    
    - Finally missing keys are detected using normal template matching and  is marked with a red rectangle around the missing area.
"""
