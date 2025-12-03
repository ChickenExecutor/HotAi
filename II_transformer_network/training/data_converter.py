import csv
import os

from training import data_dir, data_file_write_path


def convert_txt_to_pkl(txt_file_path, output_file_path):
    reader = csv.reader(open(txt_file_path, "r", encoding="utf-8"), delimiter="\t")
    data = []
    for line in reader:
        data.append([line[0], line[1]])
    with open(output_file_path.replace(".pkl", ".csv"), "w", encoding="utf-8") as output_file:
        csv.writer(output_file, delimiter=",").writerows(data)


if __name__ == '__main__':
    txt_file = os.path.join(data_dir, "deu.txt")
    convert_txt_to_pkl(txt_file, data_file_write_path)
