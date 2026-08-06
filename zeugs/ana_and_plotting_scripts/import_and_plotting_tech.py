from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from measurements import *
from statana import *
from ana_tail import *
from ana_mats import FitSubSum, build_integrand

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


import scienceplots
plt.style.use('science')
plt.rcParams.update({
    'font.size' : 10,
    'text.usetex' : True
})



latex_width_pts = 469.75502
wlatex = latex_width_pts / 72.27

golden_ratio = (5**0.5 - 1) / 2
hlatex = wlatex * golden_ratio