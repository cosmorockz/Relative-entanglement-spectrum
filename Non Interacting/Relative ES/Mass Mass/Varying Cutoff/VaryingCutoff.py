import os
import numpy as np

M = np.array([1,2,3,4,6,10,15,20])

for m in M:
	with open('Cutoff.dat','w') as MyFile:
		MyFile.write(str(int(m)))
	os.system('python MASS-MASS.py')



