import os
from typing import List


def find_files_in_dir(directory: str, search_string: str, mode: int = 2) -> List[str]:
    """Find files in a directory based on a search string and matching mode.

    Args:
        directory: The path to the directory to search in.
        search_string: The string to search for in the file names.
        mode: The matching mode. Defaults to 2.

            0: Startswith mode

            1: Endswith mode

            2: Contains mode

    Returns:
        A list of file paths whose names match the specified mode.

    """
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if mode == 0 and file.startswith(search_string):
                file_list.append(os.path.join(root, file))
            elif mode == 1 and file.endswith(search_string):
                file_list.append(os.path.join(root, file))
            elif mode == 2 and search_string in file:
                file_list.append(os.path.join(root, file))
    return file_list

