import numpy as np
import cv2
import laspy
from sklearn.cluster import DBSCAN

# Information from the header
Z_THRESHOLD = 0.0  # Separating floor and wall, adjust as needed
DOWNSAMPLE_RATE = 10  # Adjust based on desired downsampling
RESOLUTION = (1024, 1024)  # Resolution for the depth image
THRESHOLD_FACTOR = 0.1  # Adjust as needed for fine-tuned
Ceiling_min_Z = 10  # From the header, highest elevation
num_points = 4192232
min_bounds = [-1.1477099657058716, 192.084716796875, -0.761]
max_bounds = [9.949290034294128, 203.333716796875, 2.497]
density = 10308.007500947195  # points per unit volume

# Refer calculation to distance.py file
average_distance = 1.5646736758288433

# Use the distance and density for eps and min samples value
eps = 0.8 * average_distance
min_samples = max(1, int(density * (eps ** 4)))
def read_las_file(file_path):
    inFile = laspy.file.File(file_path, mode="r")
    points = np.vstack((inFile.x, inFile.y, inFile.z)).transpose()
    inFile.close()
    return points

# Downscale the point cloud
def preprocess_point_cloud(points):
    downsampled_points = points[::DOWNSAMPLE_RATE]
    return downsampled_points

# Segment floors and walls
def segment_floor_and_walls(points):
    floor_points = points[points[:, 2] < Z_THRESHOLD]
    wall_points = points[points[:, 2] >= Z_THRESHOLD]
    return floor_points, wall_points

# Generate image
def generate_depth_image(wall_points):
    min_val = np.min(wall_points, axis=0)
    max_val = np.max(wall_points, axis=0)
    normalized_points = (wall_points - min_val) / (max_val - min_val)

    resolution = np.array(RESOLUTION)
    image_points = np.zeros_like(normalized_points, dtype=np.int32)
    for i in range(normalized_points.shape[0]):
        image_points[i, 0] = int(normalized_points[i, 0] * resolution[0])
        image_points[i, 1] = int(normalized_points[i, 1] * resolution[1])

    depth_image = np.zeros(RESOLUTION, dtype=np.uint8)
    for point in image_points:
        cv2.circle(depth_image, (point[0], point[1]), 1, 255, -1)
    return depth_image

# To detect any lines
def detect_lines(depth_image):
    edges = cv2.Canny(depth_image, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=30, maxLineGap=10)
    return lines

# DBSCAN
def object_detection(floor_points, all_points):
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(all_points[:, :2])
    labels = clustering.labels_
    object_mask = (labels == -1)  # -1 label indicates outliers
    object_points = all_points[object_mask]
    return object_points

# Process file to get needed files
def process_las_file(file_path):
    points = read_las_file(file_path)
    downsampled_points = preprocess_point_cloud(points)
    floor_points, wall_points = segment_floor_and_walls(downsampled_points)
    depth_image = generate_depth_image(wall_points)
    lines = detect_lines(depth_image)
    object_points = object_detection(floor_points, downsampled_points)
    return depth_image, lines, object_points

# Final 2D floorplan
def display_floorplan(depth_image, object_points):
    floorplan_image = np.zeros_like(depth_image)
    for point in object_points:
        cv2.circle(floorplan_image, (point[0], point[1]), 1, 255, -1)
    return floorplan_image

# Loading data
file_path = 'Airbnb_1cm.las'  #
depth_image, lines, object_points = process_las_file(file_path)

cv2.imwrite('depth_image.png', depth_image)
cv2.imshow('Depth Image', depth_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
