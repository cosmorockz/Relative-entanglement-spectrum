import numpy as np
import gc
import csv
import os
import gzip
import glob
import shutil
import psutil
import logging
import warnings
import threading
import scipy as sp
import itertools
from scipy.sparse.linalg import eigsh,svds
from scipy import sparse
from Overlaps import OverlapGenerator
import concurrent.futures
from joblib import Parallel, delayed


mz = - 2
Mz = 2
cutoff = 3
tol = 1e-14

NumberofStates = 2 * cutoff * (cutoff + 1)

O = OverlapGenerator(cutoff,mz)

# num_cores = multiprocessing.cpu_count()
toll = 1e-10
maxFileSize = 1000000
warnings.filterwarnings("ignore", category=DeprecationWarning)
process = psutil.Process(os.getpid())
gcProb = 0.004
MatrixElementtol = 1e-8

DataFolder = "./BlockDiagonalData"

logging.basicConfig(filename="loggingData.log",format='%(asctime)s %(message)s', \
                      level=logging.DEBUG)
logging.getLogger('matplotlib.font_manager').disabled = True



