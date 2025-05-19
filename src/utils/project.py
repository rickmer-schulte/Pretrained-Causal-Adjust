import os

def set_root(marker="pyproject.toml"):
    """
    Set the working directory to the project root based on a specific marker file.

    Args:
        marker (str): The filename used to identify the project root. Default is "pyproject.toml".

    Returns:
        str: The absolute path to the project root.

    Raises:
        FileNotFoundError: If the marker file is not found in the current directory or any of its parents.
    """
    cur_dir = os.path.abspath(os.getcwd())
    while True:
        if os.path.exists(os.path.join(cur_dir, marker)):
            os.chdir(cur_dir)
            print(f"Set working directory to project root: {cur_dir}")
            return cur_dir
        parent = os.path.dirname(cur_dir)
        if parent == cur_dir:
            raise FileNotFoundError(f"Could not find project root with {marker}")
        cur_dir = parent