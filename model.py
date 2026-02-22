import numpy as np
import scipy

import matplotlib.pyplot as plt

def new_smoothly_broken_bilinear(x,amp,R,L,x_break,c):
  """
  Grant’s bilinear function
  :param x: keV
  :param amp: y offset
  :param R: right-side (high energy) slope
  :param L: left-side (low-energy) slope
  :param x_break: Break keV
  :param c: Transition steepness
  :returns: Energy
  """
  dx=x-x_break
  return amp+(R+L)/2*dx+(R-L)*np.log(np.cosh(c*dx))/(2*c)

def newbilinearsat(x, amp, R, L, x_break, c, x_sat, y_max=2200):
  """
  See new_smoothly_broken_bilinear
  x_sat: linear saturation (haha) energy (keV)
  """
  #return new_smoothly_broken_bilinear(np.where(x>x_sat, x_sat, x), amp, R, L, x_break, c)
  bilin = new_smoothly_broken_bilinear(x,  amp, R, L, x_break, c)
  y_sat = new_smoothly_broken_bilinear(x_sat, amp, R, L, x_break, c)
  return np.where(x<x_sat, bilin, (y_sat-y_max)/(x_sat-600)*(x-x_sat)+y_sat)
  #if x < x_sat: return new_smoothly_broken_bilinear(x,  amp, R, L, x_break, c)
  #else:
  #  y_sat = new_smoothly_broken_bilinear(xsat, amp, R, L, x_break, c)
  #  return (y_sat-y_max)/(x_sat-1000)*(x-x_sat)+y_sat

def oldbilinear(x, a, b, c, d):
  return a*x + b*(1-np.exp(-x/c)) + d



class Anchor:
 """
 Helper class to anchor a keV value at a certain ToT. Not working yet.
 """

  def __init__(self, energy, ToT, model, number=1, height=1000, spread=100):
    """
    Generates triangle-shaped anchors at a given energy
    """
    self.shift = 2000
    self.energy, self.ToT = energy, ToT
    self.model = model
    self.number = number
    self.height, self.spread = height, spread
    self.__process(self.model.pars)
    # Define the triangle in energy space and transform it to ToT space
    y = self.height - np.abs(np.arange(-self.shift, 5*self.shift) - self.energy)*self.height/self.spread
    self.y = np.where(y<0, 0, y)

  def __process(self, pars):
    # x is 0-1 MeV, transformed to ToT space
    resp = lambda x:  self.model.spfun(x, *pars[:self.model.nfpars])
    self.x = resp(np.arange(-self.shift, 5*self.shift))
    self.x2 = self.x - resp(self.energy) + self.ToT

  def chi2(self, pars):
    self.__process(pars)
    f1 = scipy.interpolate.interp1d(self.x, self.y, kind="linear")
    f2 = scipy.interpolate.interp1d(self.x2, self.y, kind="linear")
    totlist = np.arange(400, 1000, 10)
    return np.sum(np.square(f1(totlist)-f2(totlist)))


  def plot(self, *args, **kwargs):
    plt.plot(self.x, self.y, *args, **kwargs)
    plt.plot(self.x2, self.y, *args, **kwargs)



class Model:
  """
  Takes a list of event energies (in keV, no noising), a list of weights, and bins in ToT space
  Generate an model spectrum in ToT space by applying DEE to keV data and histogramming them.
  """

  def __init__(self, data, bins, spfun=oldbilinear, 
                p0=[5/2,3775/2,19,-2328/2, 10, 200, 1013], anchors=[(22, 510),(60,750)]):
    """
    :param data: np.array of dim 2xN (keV values, weights)
    :param bins: passed to np.histogram (in ToT space)
    :param spfun: spectral response function (keV -> ToT)
    :param p0: parameters of spfun + number of events, initial guess of spectral fit
    :param anchors: list of couple (energy, ToT)
    """
    self.data, self.weights = data[0], data[1]
    self.spfun = spfun
    self.nfpars = len(p0)-1
    self.pars = np.array(p0)
    self.bins = bins
    self.anchors = [Anchor(e, t, self, i+1) for i, (e,t) in enumerate(anchors)]
    self.__computeBeta(self.pars) # fills self.hist

  def getx(self):
    return np.hstack([self.xaxis]+[e.x for e in self.anchors])

  def show(self, yapix):
    """
    """
    plt.errorbar(self.xaxis, yapix, yerr=np.sqrt(np.where(yapix<0, 0, yapix)), label="data")
    plt.errorbar(self.xaxis, self.hist, yerr=np.sqrt(self.histerr), label="beta")
    plt.legend()
    plt.xlim(-200, 2500)
    plt.grid(True, "both"); plt.show()


  def __computeBeta(self, pars):
    """
    Fill self.hist with the ToT model spectrum
    """
    self.weights *= pars[-1]/self.weights.sum()
    self.pars = np.array(pars)
    self.hist, self.bins = np.histogram(self.spfun(np.random.normal(self.data, 3), *self.pars[:self.nfpars]), bins=self.bins, weights=self.weights)
    self.histerr = np.histogram(self.spfun(np.random.normal(self.data, 3), *self.pars[:self.nfpars]), bins=self.bins, weights=np.square(self.weights))[0]
    self.xaxis = .5*(self.bins[:-1]+self.bins[1:])

  def chi2(self, pars, yapix, yerr):
    self.__computeBeta(pars)#updates model parameters
    return np.sum([a.chi2(pars) for a in self.anchors])+np.sum(np.square((self.hist-yapix)/(yerr*yerr+self.histerr)))

  def __call__(self, x, *pars):
    """
    :params: See new_smoothly_broken_bilinear
    :returns: value of nearest bin center in ToT space
    """
    if np.any(self.pars != np.array(pars)):
      self.__computeBeta(pars) #Parameters changed, re-apply response function to beta spectrum
    betay = scipy.interpolate.interp1d(self.xaxis, self.hist, kind='nearest')(x) # return closest matching bin
    #center31 = self.spfun(31, *self.pars[:-3])
    #sigma31 = self.spfun(31-self.pars[-3], *self.pars[:-3])
    return betay# + self.pars[-2]*scipy.stats.norm.pdf(x, center31, sigma31)#Add 31 keV gaussian in TOT space



