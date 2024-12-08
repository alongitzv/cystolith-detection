from cgi import test
import glob, os

# Current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

print(current_dir)

current_dir = 'data/obj'

# Percentage of images to be used for the test set
percentage_test = 10;

# Ranges of test and train data
real_files = {"test": [[2, 20], [209, 227], [1126, 1142], [2021, 2047], [2297, 2299]],
              "train": [[22, 207], [229, 1124], [1144, 2019], [2051, 2250], [2253, 2293]]}

fake_files = {"test": [[1, 10], [101, 110], [201, 210], [301, 310], [401, 407]],
              "train": [[11, 100], [111, 200], [211, 300], [311, 400], [408, 500]]}

# Create and/or truncate train.txt and test.txt
file_train = open('data/train.txt', 'w')
file_test = open('data/test.txt', 'w')

# Populate train.txt and test.txt
test_flag = False

for filename in os.listdir(current_dir):
    if filename.endswith(".jpg"):
        title, ext = os.path.splitext(os.path.basename(filename))

        start = title.find("(") + 1
        end = title.find(")")
        current = int(title[start:end])

        if title.startswith("C"):
            test_flag = False
            for list in real_files["test"]:
                if current >= list[0] and current<=list[1]:
                    test_flag = True
            
            if test_flag:
                file_test.write("data/obj" + "/" + title + '.jpg' + "\n")
            else:
                file_train.write("data/obj" + "/" + title + '.jpg' + "\n")
        
        if title.startswith("S"):
            test_flag = False
            for list in fake_files["test"]:
                if current >= list[0] and current<=list[1]:
                    test_flag = True

            if test_flag:
                file_test.write("data/obj" + "/" + title + '.jpg' + "\n")
            else:
                file_train.write("data/obj" + "/" + title + '.jpg' + "\n")
