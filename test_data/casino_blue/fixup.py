import os
from pathlib import Path
import shutil

dirs = ["five", "four", "one", "six", "three", "two"]

# So bad, but whatever...
for dir in dirs:
	file_list = Path(os.path.join(dir)).glob("*.jpg")
	for file in file_list:
		file_name = os.path.basename(file)
		for other_dir in dirs:
			if other_dir != dir:
				other_file_list = Path(os.path.join(other_dir)).glob("*.jpg")
				for other_file in other_file_list:
					other_file_name = os.path.basename(other_file)
					if other_file_name == file_name and dir < other_dir:
						print("Duplicate files: {} and {}".format(file, other_file))