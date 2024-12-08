def split_data(input_file, train_output_file, val_output_file):
    # Read the input file and split it into lines
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Calculate the number of lines for validation (1/16 of total lines)
    num_val_lines = len(lines) // 20

    # Split the data into training and validation sets
    train_data = lines[num_val_lines:-num_val_lines]
    val_data = lines[:num_val_lines] + lines[-num_val_lines:]

    # Write training data to train.txt
    with open(train_output_file, 'w') as train_file:
        train_file.writelines(train_data)

    # Write validation data to val.txt
    with open(val_output_file, 'w') as val_file:
        val_file.writelines(val_data)

if __name__ == "__main__":
    input_file = "train_and_val.txt"
    train_output_file = "train.txt"
    val_output_file = "val.txt"

    split_data(input_file, train_output_file, val_output_file)
    print("Data split into train.txt and val.txt")
