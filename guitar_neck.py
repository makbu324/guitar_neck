# Downgrade to Python 3.12!

import cv2
import numpy as np
import pyautogui
import mediapipe as mp
import math
mp.drawing = mp.solutions.drawing_utils
mp.drawing_styles = mp.solutions.drawing_styles
mphands = mp.solutions.hands
hands = mphands.Hands()
def is_point_in_rectangle(x, y, x_min, y_min, x_max, y_max):
    # Check if a point (x, y) is inside the rectangle
    return x_min <= x <= x_max and y_min <= y <= y_max

def do_lines_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
    # Helper function to check if two lines intersect
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    a = (x1, y1)
    b = (x2, y2)
    c = (x3, y3)
    d = (x4, y4)

    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)

def line_intersects_rectangle(x1, y1, x2, y2, x_min, y_min, x_max, y_max):
    # Check if the line intersects with any of the rectangle's edges
    # Rectangle edges are defined by their corners
    top_left = (x_min, y_min)
    top_right = (x_max, y_min)
    bottom_left = (x_min, y_max)
    bottom_right = (x_max, y_max)

    # Check if either point of the line is inside the rectangle
    if is_point_in_rectangle(x1, y1, x_min, y_min, x_max, y_max) or \
       is_point_in_rectangle(x2, y2, x_min, y_min, x_max, y_max):
        return True

    # Check for intersection with each edge of the rectangle
    if do_lines_intersect(x1, y1, x2, y2, *top_left, *top_right):  # Top edge
        return True
    if do_lines_intersect(x1, y1, x2, y2, *top_left, *bottom_left):  # Left edge
        return True
    if do_lines_intersect(x1, y1, x2, y2, *top_right, *bottom_right):  # Right edge
        return True
    if do_lines_intersect(x1, y1, x2, y2, *bottom_left, *bottom_right):  # Bottom edge
        return True
def modify_line_to_x_bounds(x1, y1, x2, y2, x_min, x_max):
    # Ensure x1 < x2 for proper calculations
    if x1 > x2:
        x1, y1, x2, y2 = x2, y2, x1, y1

    # Calculate the slope (m) and intercept (b) of the line
    if x2 != x1:
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
    else:
        # If the line is vertical, restrict x1 and x2 to the bounds
        if x1 < x_min or x1 > x_max:
            return None  # No part of the line is within bounds
        return x1, y1, x2, y2

    # Clamp the x1 and x2 to the bounds
    new_x1 = max(x_min, x1)
    new_y1 = m * new_x1 + b
    new_x2 = min(x_max, x2)
    new_y2 = m * new_x2 + b

    return int(new_x1), int(new_y1), int(new_x2), int(new_y2)
def find_perpendicular_line(x1, y1, x2, y2, x, y, length):
    # Calculate the slope of the original line
    if x2 != x1:
        m = (y2 - y1) / (x2 - x1)  # Slope of the original line
        # Slope of the perpendicular line
        m_perp = -1 / m
        # Half-length of the perpendicular line
        half_length = length / 2
        # Calculate dx and dy
        dx = half_length / math.sqrt(1 + m_perp**2)
        dy = m_perp * dx
        
        # Ensure the perpendicular line starts on the right or lower side
        if (x2 - x1) > 0:  # Original line is moving to the right
            x1_perp = x - dx * 2
            y1_perp = y - dx * 2
            x2_perp = x + dx * 2
            y2_perp = y + dy * 2
        else:  # Original line is moving to the left
            x1_perp = x + dx * 2
            y1_perp = y + dx * 2
            x2_perp = x - dx * 2
            y2_perp = y - dy * 2
    else:
        # Original line is vertical; perpendicular line is horizontal
        half_length = length / 2
        if (y2 - y1) > 0:  # Original line is going downward
            x1_perp = x
            y1_perp = y
            x2_perp = x + length
            y2_perp = y
        else:  # Original line is going upward
            x1_perp = x
            y1_perp = y
            x2_perp = x - length
            y2_perp = y

    # Return the two endpoints of the perpendicular line
    return int(x1_perp), int(y1_perp), int(x2_perp), int(y2_perp) 
def find_points_on_line(x1, y1, x2, y2):
    pixels = []

    # Calculate differences and steps
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1

    # Error variable
    err = dx - dy

    # Loop until the end point is reached
    while True:
        pixels.append((y1, x1))  # Add the current pixel to the list
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy

    return pixels
def getboundingbox(hl):
    x_max = 0
    y_max = 0
    x_min = w
    y_min = h
    for lm in hl.landmark:
        x, y = int(lm.x * w), int(lm.y * h)
        if x > x_max:
            x_max = x
        if x < x_min:
            x_min = x
        if y > y_max:
            y_max = y
        if y < y_min:
            y_min = y
    return x_min, y_min, x_max, y_max
def nothing(x):
    pass
try:
    cv2.namedWindow("Mask 2", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Trace", cv2.WINDOW_NORMAL)
    while True:
        img = pyautogui.screenshot(region=(0,0, 1620, 1320))
        frame = np.array(img)
        h, w, c = frame.shape
        come_true = np.zeros_like(frame)
        see = np.zeros_like(frame)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        kernel = np.ones((3, 3), np.uint8)
        dilated_image = cv2.dilate(frame, kernel, iterations=1)
        edges = cv2.Canny(dilated_image, 25, 200, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 150)
        results = hands.process(frame)
        x_1, y_1, x_2, y_2 = [-1,-1,-1,-1]
        No_Two_Points = True
        if lines is not None: 
            for rho, theta in lines[:, 0]:
                if (1.5 < theta and theta < 1.6) or theta < 0.1:
                    continue
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 10000 * (-b))
                y1 = int(y0 + 10000 * (a))
                x2 = int(x0 - 10000 * (-b))
                y2 = int(y0 - 10000 * (a))
                x1, y1, x2, y2 = modify_line_to_x_bounds(x1, y1, x2, y2, 0, 1620)
                if results.multi_hand_landmarks: 
                    points = 0
                    for hl in results.multi_hand_landmarks:
                        x_min_1, y_min_1, x_max_1, y_max_1 = getboundingbox(hl)
                        cv2.rectangle(come_true, (x_min_1, y_min_1-(y_max_1-y_min_1)//2), (x_max_1, y_max_1+(y_max_1-y_min_1)//2), (255,0,0), 1)
                        if line_intersects_rectangle(x1, y1, x2, y2, x_min_1, y_min_1-(y_max_1-y_min_1)//2, x_max_1, y_max_1+(y_max_1-y_min_1)//2):
                            points += 1
                    if points > 0 and x_1 == -1:
                        x_1 = x1
                        y_1 = y1
                        x_2 = x2
                        y_2 = y2
                    if points > 1:
                        No_Two_Points = False
                        x_1 = x1
                        y_1 = y1
                        x_2 = x2
                        y_2 = y2
                        cv2.line(come_true, (x1, y1), (x2, y2), (0, 255, 0), 2) 
                        break
            
            if not(x_1 == -1):
                if No_Two_Points: # We need to rely on voting
                    cv2.line(come_true, (x_1, y_1), (x_2, y_2), (0, 255, 0), 2) 
                else:
                    x_min_1, y_min_1, x_max_1, y_max_1 = getboundingbox(results.multi_hand_landmarks[0])
                    x_min_2, y_min_2, x_max_2, y_max_2 = getboundingbox(results.multi_hand_landmarks[1])
                    x_1, y_1, x_2, y_2 = modify_line_to_x_bounds(x_1, y_1, x_2, y_2, min(x_min_1, x_max_1, x_min_2, x_max_2), 1620)
                    hand_height = (y_max_1-y_min_1)//2 
                    x1_perp, y1_perp, x2_perp, y2_perp = find_perpendicular_line(x_1, y_1, x_2, y_2, x_1, y_1, hand_height)
                    x3_perp, y3_perp, x4_perp, y4_perp = find_perpendicular_line(x_1, y_1, x_2, y_2, x_2, y_2, hand_height)
                    h_m = h 
                    w_m = w 
                    height_m = max(y1_perp, y2_perp, y3_perp, y4_perp) 
                    width_m = max(x1_perp, x2_perp, x3_perp, x4_perp) 
                    if height_m > h:
                        h_m += height_m - h + 1 
                    if width_m > w: 
                        w_m += width_m - w + 1
                    pts1 = np.float32([[x1_perp,y1_perp],[x3_perp,y3_perp],[x2_perp,y2_perp],[x4_perp,y4_perp]])
                    pts2 = np.float32([[0,0],[h_m,0],[0,hand_height],[h_m,hand_height]])
                    M = cv2.getPerspectiveTransform(pts1,pts2)
                    dst = cv2.warpPerspective(rgb,M,(h_m,hand_height))
                    cv2.imshow("Mask 2", dst) 
        
        if results.multi_hand_landmarks: 
            for hl in results.multi_hand_landmarks:
                mp.drawing.draw_landmarks(come_true,hl,mphands.HAND_CONNECTIONS) 
        
        
        1# cv2.imshow("Mask 2", edges)
        cv2.imshow("Trace", come_true)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("Stopped recording")
cv2.destroyAllWindows()
# 