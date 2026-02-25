import cv2
import numpy as np

def detect_dome_curves(image_path, threshold_factor=0.45):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 1. Smooth the image to ignore speckle noise
    # We use a larger vertical sigma to prioritize vertical consistency
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    # 2. Identify shadow columns (re-using your logic)
    vertical_profile = np.mean(blurred, axis=0)
    limit = np.median(vertical_profile) * threshold_factor
    shadow_indices = np.where(vertical_profile < limit)[0]

    result_img = img.copy()
    
    # 3. Trace the Dome Boundary
    # For each column, find the transition from bright to dark
    dome_points = []
    
    # We ignore the very top of the image where UI text ("dray") might be
    top_offset = 60 

    for x in shadow_indices:
        column = blurred[top_offset:, x]
        
        # Calculate the gradient (change in brightness)
        # We look for a sharp negative gradient (Bright -> Dark)
        gradient = np.diff(column.astype(float))
        
        # Find the pixel where the brightness drops below a threshold
        # and where the gradient is strongest (the edge of the dome)
        potential_edges = np.where((column[:-1] < 60) & (gradient < -1))[0]
        
        if len(potential_edges) > 0:
            y_coord = potential_edges[0] + top_offset
            dome_points.append((x, y_coord))
            
    # 4. Draw the dome shape using a polyline or points
    if len(dome_points) > 2:
        # Drawing as individual points to show the dome structure
        for pt in dome_points:
            cv2.circle(result_img, pt, 2, (0, 255, 0), -1)
            
    return result_img

# Execution
image_path = "./frames/Prior_Scan_zoom/00001.jpg"
final_result = detect_dome_curves(image_path)
cv2.imwrite("dome_shape_detected.png", final_result)