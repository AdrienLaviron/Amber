import numpy as np
import scipy
from scipy.optimize import curve_fit, least_squares, minimize
import scipy.stats
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
      if len(elist)>0:
        add2dict(dfdict, gethitdata(Event.GetHTAt(int(np.argmax(elist)))), mce, nhits)
    elif strategy == "all":
      for i in range(nhits):
        add2dict(dfdict, gethitdata(Event.GetHTAt(i)), mce, nhits)
    elif strategy == "single":
      if nhits == 1: add2dict(dfdict, gethitdata(Event.GetHTAt(0)), mce, nhits)
    else:
      raise RuntimeError
  return pd.DataFrame(dfdict)


if __name__ == "__main__" and False:
  # Init ROOT global variables
  initroot()
  # Get geometry
  geometry = getGeometry("ViewSiPixelDetector.geo.setup")
  geometry.ActivateNoising(False)
  # Read simulated data
  csbeta = Spectrum("spectra/Cs-137_Beta_Spectrum.csv")
  data = []; depths = []
  bins = np.arange(0, 1300, 20)
  flist = glob.glob("simulations/Cs137_d*.sim.gz")
  cmap = plt.get_cmap('jet', 200)
  for i, f in enumerate(flist):
    depth = float(".".join(f.split("_d")[-1].split(".")[:2]))*1e4
    print(i, f, depth)
    depths.append(depth)
    df = gethits(f, geometry, "max")
    df = df[((df.HTX>-1.9)&(df.HTX<-1.1))|((df.HTX>.1)&(df.HTX<.9))] # no BusBar
    #df = df[((df.HTX>-1)&(df.HTX<-.1))|((df.HTX>1)&(df.HTX<1.7))] # With BusBar
    df["WT"] = csbeta(df["MCE"]*1e-3)
    data.append(df)
    plt.hist(df.HTE, weights=df.WT, bins=bins, histtype='step', color=cmap(int(depth)))
    np.save(f"cs137d{int(depth)}m.npy", np.vstack((df.HTE, df.WT)))
  print(depths)
  norm = matplotlib.colors.Normalize(vmin=0,vmax=200)
  sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
  sm.set_array([])
  plt.colorbar(sm, ticks=np.arange(0, 251, 10), ax=plt.gca(), label="Depletion depth")
  plt.xlabel("Deposited energy (keV)")
  plt.ylabel("# Hits")
  plt.grid(True, "both"); plt.show()

if __name__ == "__main__" and False:
  # Init ROOT global variables
  initroot()
  # Get geometry
  geometry = getGeometry("ViewSiPixelDetector.geo.setup")
  geometry.ActivateNoising(False)
  # Read simulated data
  csbeta = Spectrum("spectra/Cs-137_Beta_Spectrum.csv")
  data = []; thetas = []
  bins = np.arange(0, 1300, 20)
  flist = glob.glob("simulations/Cs137_theta*.sim.gz")
  cmap = plt.get_cmap('jet', 50)
  for i, f in enumerate(flist):
    theta = float(".".join(f.split("_theta")[-1].split(".")[:2]))
    print(i, f, theta)
    thetas.append(theta)
    df = gethits(f, geometry, "max")
    df = df[((df.HTX>-1.9)&(df.HTX<-1.1))|((df.HTX>.1)&(df.HTX<.9))] # no BusBar
    #df = df[((df.HTX>-1)&(df.HTX<-.1))|((df.HTX>1)&(df.HTX<1.7))] # With BusBar
    df["WT"] = csbeta(df["MCE"]*1e-3)
    data.append(df)
    plt.hist(df.HTE, weights=df.WT, bins=bins, histtype='step', color=cmap(int(theta)))
    np.save(f"cs137theta{int(theta)}deg.npy", np.vstack((df.HTE, df.WT)))
  print(thetas)
  norm = matplotlib.colors.Normalize(vmin=0,vmax=50)
  sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
  sm.set_array([])
  plt.colorbar(sm, ticks=np.arange(0, 51, 5), ax=plt.gca(), label="Off-axis angles")
  plt.xlabel("Deposited energy (keV)")
  plt.ylabel("# Hits")
  plt.grid(True, "both"); plt.show()


if __name__ == "__main__" and False:
  # Init ROOT global variables
  initroot()
  # Get geometry
  geometry = getGeometry("ViewSiPixelDetector.geo.setup")
  geometry.ActivateNoising(False)
  # Read simulated data
  csbeta = Spectrum("spectra/Cs-137_Beta_Spectrum.csv")
  bins = np.arange(10, 1300, 10)
  hists = []
  for s in ["all", "max", "single"]:
    df = gethits("simulations/Cs137_d0.007.inc1.id1.sim.gz", geometry, s)
    df["WT"] = csbeta(df["MCE"]*1e-3)
    hists.append(np.histogram(df.HTE, weights=df.WT, bins=bins)[0])
    plt.hist(df.HTE, weights=df.WT, bins=bins, histtype="step", label=s)
  plt.legend()
  plt.yscale("log")
  plt.grid(True, "both"); plt.show()
  x = .5*(bins[:-1]+bins[1:])
  plt.plot(x, hists[1]/hists[0])
  plt.plot(x, hists[2]/hists[0])
  plt.grid(True, "both"); plt.show()

from model import *

if __name__ == "__main__" and True:
  # Get data
  apix = pd.read_csv("data/20260105-172323_matched.csv")
  yapix, bins = np.histogram(apix.row_tot[(apix.layer==1)&(apix.chipID==0)&(apix.row==19)&(apix.col==19)], np.arange(0, 4000, 100))
  yerr = np.sqrt(yapix)
  yerr = np.where(yerr<1, 1, yerr)

if __name__ == "__main__" and False:
  depths = [50, 70, 90, 110, 130]
  #depths = [70]
  p0 = [980, 11, 13, 60, .05, 300, np.sum(yapix)]
  fitbounds = [(10,1600), (.1, 5), (1, 20), (30, 90), (.003, 5), (180, 300), (50,4000)]

  for dnumber, depth in enumerate(depths):
    print(f"{depth} um")
    model=Model(np.load(f"cs137d{depth}m.npy"), bins, spfun=newbilinearsat, p0=p0)

    if len(depths)<3:
      factors = np.arange(.2, 5, .1)
      chi2 = np.empty((len(p0), len(factors)))
      for i in range(len(model.pars)):
        pars = [e for e in p0]
        for j, p in enumerate(factors):
          pars[i] = p0[i]*p
          chi2[i][j] = model.chi2(pars, yapix, yerr)
        plt.plot(factors, chi2[i], label=f"{i}")
      plt.legend()
      plt.grid(True, "both"); plt.show()

    res = minimize(model.chi2, p0, args=(yapix, yerr), bounds=fitbounds)
    print(res.x)
    for a in model.anchors: a.plot()
    model.show(yapix)

if __name__ == "__main__" and True:
  p0 = [980, 5, 13, 50, .05, 300, np.sum(yapix)]
  model=Model(np.load(f"cs137d70m.npy"), bins, spfun=newbilinearsat, p0=p0)
  res = minimize(lambda x: model.anchors[0].chi2((980, x[0], x[1], 45, .04, 300, 1100))+model.anchors[1].chi2((980, x[0], x[1], 45, .04, 300, 1100)), (5, 12))
  #factors = np.array([.2, .5, 1, 2, 5])
  #for i in range(4):
  #  print(f"Par. {i}")
  #  pars = [e for e in p0]
  #  plt.plot(model.anchors[0].x2, model.anchors[0].y, "C0-")
  #  plt.plot(model.anchors[1].x2, model.anchors[1].y, "C0--")
  #  for j, p in enumerate(factors):
  #    pars[i] = p0[i]*p
  #    model.chi2(pars, yapix, yerr)
  #    plt.plot(model.anchors[0].x, model.anchors[0].y, f"C{j+1}-")
  #    plt.plot(model.anchors[1].x, model.anchors[1].y, f"C{j+1}--")
  #    #model.show(yapix)
  #  plt.xlim(-200, 3000)
  #  plt.grid(True, "both"); plt.show()

if __name__ == "__main__" and False:
  # Get data
  apix = pd.read_csv("data/20260105-172323_matched.csv")
  yapix, bins = np.histogram(apix.row_tot[(apix.layer==1)&(apix.chipID==0)&(apix.row==19)&(apix.col==19)], np.arange(0, 4200, 100))

  depths = np.arange(10, 200, 10)
  depths = [50, 60, 70, 80, 90]
  depths = [70]
  energies = np.array([22, 60])
  ToTs = np.array([510, 750])
  dchi2 = np.empty(len(depths))
  #p0 = [2 , 2000 , 22 , -1200, 10, 200, np.sum(yapix)]
  #fitbounds = ([1, 1800, 5, -1800, 50], [10, 3000, 50, -600, 1500])
  #p0 = [80, 0.19*50, 2.29*50, 83, 0.04, 10, 200, np.sum(yapix)]
  #p0 = [987, 4, 12, 40, .04, 200, 2, 200, 2*np.sum(yapix)]
  #fitbounds = ([100, .1, 10, 30, .03, 180, .2, 10, 50], [1600, 200, 200, 50, .05, 300, 20, 1500, 4000])
  p0 = [980, 2, 11, 50, .05, 200, np.sum(yapix)]
  fitbounds = ([10, .1, 1, 30, .03, 180, 50], [1600, 200, 200, 50, 5, 300, 4000])

  for dnumber, depth in enumerate(depths):
    #model = Model(np.load(f"cs137d{depth}m.npy"), bins, spfun=oldbilinear, p0=p0)
    #model=Model(np.load(f"cs137d{depth}m.npy"), bins, spfun=new_smoothly_broken_bilinear, p0=p0)
    model=Model(np.load(f"cs137d{depth}m.npy"), bins, spfun=newbilinearsat, p0=p0)

    # Second attempt, find chi squared minima by hand
    if False:
      factors = np.arange(.2, 5, .1)
      chi2 = np.empty((len(p0), len(factors)))
      for i in range(len(model.pars)):
        pars = [e for e in p0]
        for j, p in enumerate(factors):
          pars[i] = p0[i]*p
          model(100, *pars) #recompute hist
          chi2[i][j] = np.sum(np.square(yapix - model(model.xaxis, *model.pars)))
        plt.plot(factors, chi2[i], label=f"{i}")
      plt.legend()
      plt.grid(True, "both"); plt.show()

    # First attempt at general fit
    yerr = np.sqrt(yapix)
    #plt.errorbar(model.xaxis, yapix, yerr = yerr)
    #plt.plot(model.xaxis, model.hist)
    #plt.grid(True, "both"); plt.show()
    #popt, pcov = curve_fit(model, model.xaxis, yapix, p0=model.pars)
    popt, pcov = curve_fit(model, model.xaxis, yapix, sigma=np.where(yerr<1, 1, yerr), absolute_sigma=True, p0=p0, bounds=fitbounds)
    #results = least_squares(lambda params, x, y: y - model(x, *params), model.pars, args=(model.xaxis, yapix))
    print(f"d={depth} um")
    for e, tot in zip(energies, ToTs):
      print(f"{e} keV -> {model.spfun(e, *popt[:model.nfpars])} ToT (expected {tot})")
    print(popt)
    model.show(yapix, popt)
    #plt.errorbar(model.xaxis, yapix, yerr=yerr)
    #plt.plot(model.xaxis, model.hist)
    #plt.grid(True, "both"); plt.show()
    dchi2[dnumber] = np.sum(np.square(model.spfun(energies, *popt[:model.nfpars])-ToTs))
    #print(dchi2[dnumber])

  plt.plot(depths, dchi2)
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
