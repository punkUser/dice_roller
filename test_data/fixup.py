import os
from pathlib import Path

fileList = Path(os.path.join('hit')).glob('*.jpg')
for file in fileList:
	fileName = os.path.basename(file)
	
	if Path(os.path.join('blank', fileName)).exists() or Path(os.path.join('crit', fileName)).exists() or Path(os.path.join('focus', fileName)).exists():
		print("Duplicate: {}".format(fileName))
		os.remove(file)