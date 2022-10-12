import os
import numpy as np

M = np.linspace(0,4,11)

for m in M:
	with open('Splittng Modes Pairing.dat','w') as MyFile:
		MyFile.write(str(np.round(m,1)))
	os.system('python MASSMASS.py')



