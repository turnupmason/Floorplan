import numpy as np
import laspy

def read_las_file(file_path):
    inFile = laspy.file.File(file_path, mode="r")
    points = np.vstack((inFile.x, inFile.y, inFile.z)).transpose()
    inFile.close()
    return points

def calculate_average_distance(points):
    num_points = len(points)
    total_distance = 0

    for i in range(num_points):
        distance = np.linalg.norm(points - points[i], axis=1)
        total_distance += np.sum(distance)

    average_distance = total_distance / (num_points * (num_points - 1))
    return average_distance

if __name__ == "__main__":
    # LAS file path
    las_file_path = 'Airbnb_1cm.las'

    # Read LAS file
    points = read_las_file(las_file_path)

    # Consider only the first 1,000,000 points
    points = points[:1000000]

    # Calculate average point distance
    average_distance = calculate_average_distance(points)

    # Print the average point distance
    print("Average Point Distance:", average_distance)
