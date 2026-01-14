import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors

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
      add2dict(dfdict, gethitdata(Event.GetHTAt(np.argmax(elist))), mce, nhits)
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
  print(depths)
  norm = matplotlib.colors.Normalize(vmin=0,vmax=200)
  sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
  sm.set_array([])
  plt.colorbar(sm, ticks=np.arange(0, 251, 10), ax=plt.gca(), label="Depletion depth")
  plt.xlabel("Deposited energy (keV)")
  plt.ylabel("# Hits")
  plt.grid("both", True); plt.show()

 
if __name__ == "__main__" and True:
  apix = pd.read_csv()



  #df = gethits("simulations/Cs137.inc1.id1.sim.gz", geometry)
  #nobb = df[((df.HTX>-1.9)&(df.HTX<-1.1))|((df.HTX>.1)&(df.HTX<.9))]
  #wibb = df[((df.HTX>-1)&(df.HTX<-.1))|((df.HTX>1)&(df.HTX<1.7))]
  # Apply beta spectrum
  #csbeta = Spectrum("spectra/Cs-137_Beta_Spectrum.csv")
  #df["WT"] = csbeta(df["MCE"]*1e-3)
  #addwt(df, csbeta)




