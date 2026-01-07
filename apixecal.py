import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import ROOT as M

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
    print("Unable to load geometry " + GeometryName + " - Aborting!")
    quit()

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
    else:
      raise RuntimeError
  return pd.DataFrame(dfdict)

def betasp(E):
  pass#Do we even know that?

def addwt(df):
  """
  """
  df.addcol("wt")


if __name__ == "__main__":
  initroot()
  geometry = getGeometry("ViewSiPixelDetector.geo.setup")
  geometry.ActivateNoising(False)
  df = gethits("ecalsim.inc2.id1.sim.gz", geometry)
  nobb = df[((df.HTX>-1.9)&(df.HTX<-1.1))|((df.HTX>.1)|(df.HTX<.9))]
  wibb = df[((df.HTX>-1)&(df.HTX<-.1))|((df.HTX>1)&(df.HTX<1.7))]

