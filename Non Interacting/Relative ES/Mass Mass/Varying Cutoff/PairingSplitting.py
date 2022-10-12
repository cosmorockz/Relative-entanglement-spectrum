import os
import numpy as np

M = np.linspace(2,10,9)

for m in M:
	with open('Cutoff.dat','w') as MyFile:
		MyFile.write(str(np.round(m,1)))
	os.system('python MASSMASS.py')



