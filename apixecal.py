import numpy as np
import scipy
from scipy.optimize import curve_fit, least_squares
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.colors

import glob

import ROOT as M

# Class for modelizing beta spectrum (callable)
class Spectrum:
  def __init__(self, file):
    self.data = pd.read_csv(file)
    self.energies = self.data["Energy [MeV]"]
    self.proba = self.data["Probability density [#/MeV/nt]"]

  def __call__(self, E):
    return np.interp(E, self.energies, self.proba)


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

def oldbilinear(x, a, b, c, d):
  return a*x + b*(1-np.exp(-x/c)) + d


# MEGAlib data management functions
def initroot():
  M.gSystem.Load("$MEGALIB/lib/libMEGAlib.so")
  G = M.MGlobal()
  G.Initialize()

def getGeometry(GeometryName):
  Geometry = M.MDGeometryQuest()
  if Geometry.ScanSetupFile(M.MString(GeometryName)) == True:
    print("Geometry " + GeometryName + " loaded!")
    return Geometry
  else:
    raise RuntimeError("Unable to load geometry " + GeometryName + " - Aborting!")

def gethitdata(hit):
  """
  :param hit: MEGAlib MSimHT
  :returns: tuple, x,y,z,E of hit
  """
  p = hit.GetPosition()
  return p.GetX(), p.GetY(), p.GetZ(), hit.GetEnergy()

def add2dict(dfdict, htdat, mce, nhits):
  dfdict["HTX"].append(htdat[0])
  dfdict["HTY"].append(htdat[1])
  dfdict["HTZ"].append(htdat[2])
  dfdict["HTE"].append(htdat[3])
  dfdict["MCE"].append(mce)
  dfdict["NHT"].append(nhits)

def gethits(fname, geometry, strategy="all"):
  """
  :param fname: file name
  :param geometry: MDGeometryQuest
  :param strategy: str, how to deal with multiple hits, "all" or "max" (max energy)
  :retuns: Pandas Dataframe
  """
  Reader = M.MFileEventsSim(geometry)
  Reader.Open(fname)
  dfdict = {"HTX":[], "HTY":[], "HTZ":[], "HTE":[], "MCE":[], "NHT":[]}
  while(Event := Reader.GetNextEvent()):
    M.SetOwnership(Event, True)
    nhits = Event.GetNHTs()
    mce = Event.GetIAAt(0).GetEnergy()
    if strategy == "max":
      elist = [Event.GetHTAt(i).GetEnergy() for i in range(nhits)]
      add2dict(dfdict, gethitdata(Event.GetHTAt(np.argmax(elist))), mce, nhits)
    elif strategy == "all":
      for i in range(nhits):
        add2dict(dfdict, gethitdata(Event.GetHTAt(i)), mce, nhits)
    elif strategy == "single":
      if nhits == 1: add2dict(dfdict, gethitdata(Event.GetHTAt(0)), mce, nhits)
    else:
      raise RuntimeError
  return pd.DataFrame(dfdict)


if __name__ == "__main__" and True:
  # Init ROOT global variables
  initroot()
  # Get geometry
  geometry = getGeometry("ViewSiPixelDetector.geo.setup")
  geometry.ActivateNoising(False)
  # Read simulated data
  csbeta = Spectrum("spectra/Cs-137_Beta_Spectrum.csv")
  data = []; depths = []
  bins = np.arange(0, 1300, 20)
  flist = glob.glob("simulations/Cs137_*.sim.gz")
  cmap = plt.get_cmap('jet', 200)
  for i, f in enumerate(flist):
    depth = float(".".join(f.split("_d")[-1].split(".")[:2]))*1e4
    print(i, f, depth)
    depths.append(depth)
    df = gethits(f, geometry)
    #df = df[((df.HTX>-1.9)&(df.HTX<-1.1))|((df.HTX>.1)&(df.HTX<.9))] # no BusBar
    df = df[((df.HTX>-1)&(df.HTX<-.1))|((df.HTX>1)&(df.HTX<1.7))] # With BusBar
    df["WT"] = csbeta(df["MCE"]*1e-3)
    data.append(df)
    plt.hist(df.HTE, weights=df.WT, bins=bins, histtype='step', color=cmap(int(depth)))
    np.save(f"cs137{depth}m.npy", np.vstack(df.HTE, df.WT))
  print(depths)
  norm = matplotlib.colors.Normalize(vmin=0,vmax=200)
  sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
  sm.set_array([])
  plt.colorbar(sm, ticks=np.arange(0, 251, 10), ax=plt.gca(), label="Depletion depth")
  plt.xlabel("Deposited energy (keV)")
  plt.ylabel("# Hits")
  plt.grid(True, "both"); plt.show()


class Model:

  def __init__(self, data, bins, spfun=oldbilinear, p0=[5/2,3775/2,19,-2328/2, 1013]):
    """
    :param data: np.array of dim 2xN (keV values, weights)
    :param bins: passed to np.histogram
    :param spfun: spectral response function (keV -> ToT)
    :param p0: parameters of spfun + number of events, initial guess of spectral fit
    """
    self.data, self.weights = data[0], data[1]
    self.spfun = spfun
    self.pars = np.array(p0)
    self.bins = bins
    self.applyDE() # fills self.hist

  def recomputeModel(self, pars):
    """
    Fill self.hist with the ToT model spectrum
    """
    self.weights *= pars[-1]/self.weights.sum()
    self.pars = np.array(pars)
    self.hist, self.bins = np.histogram(self.spfun(self.data, *self.pars[:-1]), bins=self.bins, weights=self.weights)
    self.xaxis = .5*(self.bins[:-1]+self.bins[1:])

  def __call__(self, x, *pars):
    """
    :params: See new_smoothly_broken_bilinear
    :returns: value of nearest bin center in ToT space
    """
    if np.any(self.pars != np.array(pars)):
      self.recomputeModel(pars) #Parameters changed, re-apply response function
    return scipy.interpolate.interp1d(self.xaxis, self.hist, kind='nearest')(x) # return closest matching bin


 
if __name__ == "__main__" and False:
  # Get data
  apix = pd.read_csv("data/20260105-172323_matched.csv")
  yapix, bins = np.histogram(apix.row_tot[(apix.layer==1)&(apix.chipID==0)&(apix.row==19)&(apix.col==19)], np.arange(0, 4200, 100))
  model = Model(np.load("cs137hte70m.npy"), bins, spfun=oldbilinear, p0=[2 , 2000 , 22 , -1200, np.sum(yapix)])
  #model = Model(np.load("cs137hte70m.npy"), bins, np.sum(yapix), spfun=new_smoothly_broken_bilinear, p0=[80, 0.19*50, 2.29*50, 83, 0.04])

  # Second attempt, find chi squared minima by hand
  #p0 = [2 , 2000 , 22 , -1200, np.sum(yapix)]
  #factors = np.arange(.2, 5, .1)
  #chi2 = np.empty((len(p0), len(factors)))
  #for i in range(len(model.pars)):
  #  for j, p in enumerate(factors):
  #    pars = [e for e in p0]
  #    pars[i] = p0[i]*p
  #    model(100, *pars) #recompute hist
  #    chi2[i][j] = np.sum(np.square(yapix - model.hist))
  #  plt.plot(factors, chi2[i], label=f"{i}")
  #plt.legend()
  #plt.grid(True, "both"); plt.show()

  # First attempt at general fit
  yerr = np.sqrt(yapix)
  #plt.errorbar(model.xaxis, yapix, yerr = yerr)
  #plt.plot(model.xaxis, model.hist)
  #plt.grid(True, "both"); plt.show()
  fitbounds = ([1, 1800, 5, -1300, 50], [10, 2400, 50, -720, np.inf])
  #popt, pcov = curve_fit(model, model.xaxis, yapix, p0=model.pars)
  popt, pcov = curve_fit(model, model.xaxis, yapix, sigma=np.where(yerr<1, 1, yerr), absolute_sigma=True, p0=model.pars, bounds=fitbounds)
  #results = least_squares(lambda params, x, y: y - model(x, *params), model.pars, args=(model.xaxis, yapix))
  plt.errorbar(model.xaxis, yapix, yerr=yerr)
  plt.plot(model.xaxis, model.hist)
  plt.grid(True, "both"); plt.show()

  #df = gethits("simulations/Cs137.inc1.id1.sim.gz", geometry)
  #nobb = df[((df.HTX>-1.9)&(df.HTX<-1.1))|((df.HTX>.1)&(df.HTX<.9))]
  #wibb = df[((df.HTX>-1)&(df.HTX<-.1))|((df.HTX>1)&(df.HTX<1.7))]
  # Apply beta spectrum
  #csbeta = Spectrum("spectra/Cs-137_Beta_Spectrum.csv")
  #df["WT"] = csbeta(df["MCE"]*1e-3)
  #addwt(df, csbeta)

def testpar(parn, factors = np.arange(-2, 2.1, .2), p0=[5/2,3775/2,19,-2328/2]):
  plt.errorbar(model.xaxis, yapix, yerr=yerr)
  cmap=plt.get_cmap("jet", len(factors))
  for i, fact in enumerate(factors):
    pars = [e for e in p0]
    pars[parn] = pars[parn]*fact
    print(pars)
    plt.plot(model.xaxis, model(model.xaxis, *pars), color=cmap(i))
  norm = matplotlib.colors.Normalize(vmin=np.min(factors),vmax=np.max(factors))
  sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
  sm.set_array([])
  plt.colorbar(sm, ticks=factors, ax=plt.gca(), label=f"factor on par#{parn}")
  plt.grid(True, "both"); plt.show()
