import os
import cv2
import re
import numpy as np
import time


# Constants
STEP_SIZE = 40
RECT_WIDTH, RECT_HEIGHT = 50, 50

def create_directories():
    os.makedirs('images/uploads', exist_ok=True)
    os.makedirs('images/processed', exist_ok=True)

def stitch_images(upload_folder):
    image_files = [os.path.join(upload_folder, file) for file in os.listdir(upload_folder) if re.search(r'\.(jpg|jpeg|png)$', file)]
    images = [cv2.imread(img) for img in image_files]
    
    if len(images) == 0:
        raise Exception("No images found for stitching")

    stitcher = cv2.Stitcher_create()
    status, stitched_image = stitcher.stitch(images)
    
    if status == cv2.Stitcher_OK:
        stitched_image_path = os.path.join(upload_folder, 'stitched_image.jpg')
        cv2.imwrite(stitched_image_path, stitched_image)
        return stitched_image_path
    else:
        raise Exception(f"Error in stitching images: {status}")

def extract_coordinates(image_path):
    main_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    sub_image_path = 'images/uploads/crop2.png'
    
    if not os.path.exists(sub_image_path):
        raise Exception("Sub-image not found")

    sub_image = cv2.imread(sub_image_path, cv2.IMREAD_GRAYSCALE)
    main_image_gray = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(sub_image, None)
    kp2, des2 = sift.detectAndCompute(main_image_gray, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        h, w = sub_image.shape
        corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        transformed_corners = cv2.perspectiveTransform(corners, M)

        main_image_with_rectangle = main_image.copy()
        cv2.polylines(main_image_with_rectangle, [np.int32(transformed_corners)], True, (0, 0, 255), 6, cv2.LINE_AA)
        
        drone_position = tuple(map(int, np.mean(transformed_corners, axis=0)[0]))

        rect_width, rect_height = 50, 50
        rect_tl = (drone_position[0] - rect_width // 2, drone_position[1] - rect_height // 2)
        rect_br = (drone_position[0] + rect_width // 2, drone_position[1] + rect_height // 2)
        cv2.rectangle(main_image_with_rectangle, rect_tl, rect_br, (255, 0, 0), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(main_image_with_rectangle, f'Drone Position: {drone_position}', (10, 30), font, 1.5, (0, 0, 255), 4, cv2.LINE_AA)

        output_image_path = 'images/processed/output_image.jpg'
        cv2.imwrite(output_image_path, main_image_with_rectangle)
        
        return drone_position
    else:
        raise Exception("Cropped portion not found.")

def move_towards_destination(current_coords, dest_coords, step_size=STEP_SIZE):
    """Move towards the destination by a given step size."""
    direction_vector = np.array(dest_coords) - np.array(current_coords)
    direction_unit_vector = direction_vector / np.linalg.norm(direction_vector)
    new_coords = np.array(current_coords) + step_size * direction_unit_vector
    return tuple(map(int, new_coords))

def visualize_drone_state(image, drone_position, step, rect_size=RECT_WIDTH, special=False, special_rect_size=300):
    """Visualize the current state of the drone in the main image and save it to a file."""
    image_with_drone = image.copy()
    cv2.rectangle(image_with_drone, (drone_position[0] - rect_size // 2, drone_position[1] - rect_size // 2),
                  (drone_position[0] + rect_size // 2, drone_position[1] + rect_size // 2), (255, 255, 255), 2)
    if special:
        cv2.rectangle(image_with_drone, (drone_position[0] - special_rect_size // 2, drone_position[1] - special_rect_size // 2),
                      (drone_position[0] + special_rect_size // 2, drone_position[1] + special_rect_size // 2), (0, 0, 255), 10)
    cv2.putText(image_with_drone, f'Step {step}: Drone Position: {drone_position}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    output_path = os.path.join('images/processed', f'output_image_step_{step}.jpg')
    cv2.imwrite(output_path, image_with_drone)
    print(f"Image saved: {output_path}")
    return output_path

def check_destination_reached(current_coords, dest_coords):
    """Check if the destination is reached."""
    return np.linalg.norm(np.array(current_coords) - np.array(dest_coords)) < STEP_SIZE

def generate_drone_images(start_coords,special_rect_size=300):
    start_x, start_y = start_coords

    # Load the destination coordinates from crop.png
    main_image_path = 'images/uploads/stitched_image.jpg'
    sub_image_path = 'images/uploads/crop.png'
    main_image = cv2.imread(main_image_path, cv2.IMREAD_GRAYSCALE)
    sub_image = cv2.imread(sub_image_path, cv2.IMREAD_GRAYSCALE)

    # Initialize the SIFT detector
    sift = cv2.SIFT_create()

    # Find keypoints and descriptors in both images
    kp1, des1 = sift.detectAndCompute(sub_image, None)
    kp2, des2 = sift.detectAndCompute(main_image, None)

    # Initialize a Brute-Force Matcher
    bf = cv2.BFMatcher()

    # Match descriptors using KNN (K-Nearest Neighbors)
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test to find good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Extract location of good matches
    points = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

    # Assume the first good match is the destination
    sub_image_destination = tuple(map(int, points[0]))

    # Main loop to move towards sub_image destination
    current_coordinates = (start_x, start_y)
    visualization_step = 0

    # Variables to store image paths
    dest_img_path = None
    return_img_path = None

    while not check_destination_reached(current_coordinates, sub_image_destination):
        current_coordinates = move_towards_destination(current_coordinates, sub_image_destination)
        visualization_step += 1
        image_path = visualize_drone_state(main_image, current_coordinates, visualization_step)
        time.sleep(0.1)

    print("Reached sub image coordinates.")
    dest_img_path = visualize_drone_state(main_image, current_coordinates, visualization_step, special=True,special_rect_size=special_rect_size)  # Draw special rectangle  # Save the path of the image when the sub image destination is reached

    # Loop to move back to starting point
    while not check_destination_reached(current_coordinates, (start_x, start_y)):
        current_coordinates = move_towards_destination(current_coordinates, (start_x, start_y))
        visualization_step += 1
        image_path = visualize_drone_state(main_image, current_coordinates, visualization_step)
        time.sleep(0.1)

    print("Returned to starting point.")
    return_img_path = visualize_drone_state(main_image, current_coordinates, visualization_step, special=True, special_rect_size=special_rect_size)  # Draw special rectangle
  # Save the path of the image when the starting point is reached

    print(f"Destination image path: {dest_img_path}")
    print(f"Return image path: {return_img_path}")

    return dest_img_path, return_img_path

# # Ensure the functions are not executed when imported
# if __name__ == "__main__":
#     # Example usage
#     generate_drone_images((0, 0))
