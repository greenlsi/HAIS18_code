import os

def write_to_file(data_vector, path, mode="a"):
    check_path(path, mode)
    try:
        output_file = open(path, mode)
    except IOError:
        output_file = open(path, 'w')

    size = len(data_vector)

    for index, item in enumerate(data_vector):
        if index == size-1:
            output_file.write(str("{}".format(item)))
        else:
            output_file.write(str("{}".format(item)) + ",")

    output_file.write("\n")
    output_file.close()

def check_path(path, mode):
    directory = path[:path.rfind("/")]
    if mode == "a" or mode == "w":
        if not os.path.isdir(directory):
            os.mkdir(directory)
