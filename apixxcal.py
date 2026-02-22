import numpy as np
import scipy
from scipy.optimize import curve_fit, least_squares, minimize
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.colors

from model import *


class AnalyticModel:
  """
  Takes a functional pdf of events in keV space, applies DEE and does histograming in ToT space
  """

  def __init__(self, xaxis, pdf, spfun=new_smoothly_broken_bilinear, p0=None):
    """
    :param xaxis: list of keV
    :param pdf: list of n events
    """
    self.xaxis = xaxis
    self.pdf = pdf
    self.spfun = spfun
    self.__applyDEE(np.array(p0))

  def __applyDEE(self, pars):
    """
    Fills self.tot??
    """
    self.pars = pars
    self.totaxis = self.spfun(self.xaxis, *self.pars)

  def show(self, data=None):
    """
    :param data: couple (hist, bin)
    """
    if data is None:
      plt.plot(self.xaxis, self.pdf)
    if type(data)==tuple and len(data) == 2:
      plt.plot(self.totaxis, self.pdf, label="model")
      x = .5*(data[1][:-1] + data[1][1:])
      yerr = np.sqrt(data[0])
      yerr = np.where(yerr<1, 1, yerr)
      plt.errorbar(x, data[0], yerr=yerr, label="data")
      plt.legend()
    plt.grid(True, "both"); plt.show()

  def __call__(self, x, *pars):
    """
    """
    if np.any(self.pars != np.array(pars)):
      self.__applyDEE(pars)
    return scipy.interpolate.interp1d(self.totaxis, self.pdf, kind="linear")(x)


if __name__ == "__main__" and 0:
  apix = pd.read_csv("~/astropix-data/20250729-174419_matched.csv")
  yapix, bins = np.histogram(apix.row_tot[(apix.layer==1)&(apix.chipID==0)&(apix.row==19)&(apix.col==18)], np.arange(0, 4000, 20))
  yerr = np.sqrt(np.where(yapix>0, yapix, 1))
  xaxis = .5*(bins[:-1]+bins[1:])
  plt.errorbar(xaxis, yapix, yerr=yerr)
  plt.grid(True, "both"); plt.show()

if __name__ == "__main__" and 0:
  gauss = scipy.stats.norm.pdf
  efficiency = lambda eff, att, ddepth: (1-np.exp(-eff*2.33*ddepth))*np.exp(-att*1.2*.1)#Assuming 1 mm of PVC (C2H3Cl)n as absorber and ddepth depletion depth
  line = lambda x, E, width, I, efficiency: I*gauss(x, E, width)*efficiency
  pdf = lambda x, ddepth: line(x, 13.9, 2, .37, efficiency(12.52, 12.69, ddepth)) + \
                          line(x, 26.3, 2, .0227, efficiency(1.899, 2, ddepth)) + \
                          line(x, 59.54, 2, .359, efficiency(.2851, .3039, ddepth))
  kevaxis = np.arange(200)
  p0 = [1, 1, 1, 1, 1, 1e5]
  model = AnalyticModel(kevaxis, pdf(kevaxis, 0.01), p0=p0)
  model.show()

def gaussm(x, x0, h, sigma):
  return h*np.exp( - (x-x0)**2 /2 /sigma**2 )

def fit1gauss(x, data):
  pass

if __name__ == "__main__" and 1:
  pass

