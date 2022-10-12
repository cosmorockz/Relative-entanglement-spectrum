import os
import numpy as np

M = np.array([-20,-10,-6,-4,-3,-2,-1,0,1,2,3,4,6])

for m in M:
	with open("Upper Half Mass.dat",'w') as MyFile:
		MyFile.write(str(np.round(m,1)))
	os.system('python MASSMASS.py')



