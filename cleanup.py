import os
import shutil

# List of file patterns and directories to remove
REMOVE_FILES = ['*.pyc', '*.log', '*.tmp']
REMOVE_DIRS = ['__pycache__', 'build', 'dist', '.pytest_cache']

def remove_files(root_dir, patterns):
    for dirpath, _, filenames in os.walk(root_dir):
        for pattern in patterns:
            for filename in filenames:
                if filename.endswith(pattern.lstrip('*')):
                    file_path = os.path.join(dirpath, filename)
                    try:
                        os.remove(file_path)
                        print(f"Removed file: {file_path}")
                    except Exception as e:
                        print(f"Error removing file {file_path}: {e}")

def remove_dirs(root_dir, dirs):
    for dirpath, dirnames, _ in os.walk(root_dir):
        for dirname in dirnames:
            if dirname in dirs:
                dir_to_remove = os.path.join(dirpath, dirname)
                try:
                    shutil.rmtree(dir_to_remove)
                    print(f"Removed directory: {dir_to_remove}")
                except Exception as e:
                    print(f"Error removing directory {dir_to_remove}: {e}")

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    remove_files(project_root, REMOVE_FILES)
    remove_dirs(project_root, REMOVE_DIRS)
    print("Cleanup completed.")