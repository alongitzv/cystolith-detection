import os
import statistics

def extract_dimensions_from_file(file_path):
    x_values = []
    y_values = []

    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if ("fake_cystolith:" in line) or ("real_cystolith:" in line):
                line = line[:-2]
                parts = line.split()
                left_x_index = parts.index("(left_x:")
                top_y_index = parts.index("top_y:")
                width_index = parts.index("width:")
                height_index = parts.index("height:")
                x = int(parts[width_index + 1])
                y = int(parts[height_index + 1])
                x_values.append(x)
                y_values.append(y)
                if x>512 or y>512:
                    print(file_path, ": x - ", x, ", y - ", y)

    return x_values, y_values

def get_max_avg_median_dimensions_in_directory(directory):
    all_x_values = []
    all_y_values = []

    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            x_values, y_values = extract_dimensions_from_file(file_path)
            all_x_values.extend(x_values)
            all_y_values.extend(y_values)

    max_x = max(all_x_values) if all_x_values else 0
    max_y = max(all_y_values) if all_y_values else 0
    avg_x = statistics.mean(all_x_values) if all_x_values else 0
    avg_y = statistics.mean(all_y_values) if all_y_values else 0
    median_x = statistics.median(all_x_values) if all_x_values else 0
    median_y = statistics.median(all_y_values) if all_y_values else 0

    return max_x, max_y, avg_x, avg_y, median_x, median_y

directory = './output'
max_x, max_y, avg_x, avg_y, median_x, median_y = get_max_avg_median_dimensions_in_directory(directory)
print(f"Max X: {max_x}, Max Y: {max_y}")
print(f"Avg X: {avg_x}, Avg Y: {avg_y}")
print(f"Median X: {median_x}, Median Y: {median_y}")