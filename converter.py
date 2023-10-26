from util import convert_json_document_matrix
from pathlib import Path
import pickle
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Convert files")
    parser.add_argument("-s", "--source_file", help="source file")
    parser.add_argument("-d", "--dest_file_path", default="sample_data",
                        help="destination folder of new file")
    parser.add_argument("-f", "--format", help="format of the source file",
                        default="json")
    parser.add_argument("-n", "--name", help="name of the destination file")

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_arguments()
    if args.format == "json":
        matrix = convert_json_document_matrix(args.source_file)
        dest_path = Path(args.dest_file_path)
        name = args.name
        if ".pkl" not in name:
            name = name +".pkl"
        full_dest_path = dest_path / name
        with open(full_dest_path, "wb") as f:
            pickle.dump(matrix, f)

    else:
        print("Other formats not yet supported")
