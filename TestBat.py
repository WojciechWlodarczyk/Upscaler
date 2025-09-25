import sys

if __name__ == "__main__":
    print("Argumenty:", sys.argv)
    if len(sys.argv) > 2:
        folder_path = sys.argv[1]
        pt_file = sys.argv[2]
        print("Folder .bat:", folder_path)
        print("Plik .pt:", pt_file)