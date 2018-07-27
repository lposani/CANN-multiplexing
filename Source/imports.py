import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
import random
import time
from scipy.spatial.distance import cdist
from numpy.matlib import repmat
import os
import pickle
import threading
import scipy as sp
import scipy.optimize
from pylab import cm
from matplotlib.colors import LogNorm
matplotlib.rcParams.update({'font.size': 14})
defC = plt.rcParams['axes.prop_cycle'].by_key()['color']

# matplotlib.use('Agg')