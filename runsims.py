#import pathlib
#import shutil
#import os
#import glob
import time
import subprocess
import argparse
import numpy as np

def cgSimName(f, suffix):
  """
  :param f: str, path to source file
  :param suffix: suffix to add to the simulation file generated
  :returns: temporary source file name
  """
  tmpf = f"tmp/tmp_{time.time()}.source"
  with open(f, "r") as inf, open(tmpf, "w") as outf:
    lines = inf.read().split("\n")
    for line in lines:
      if line.startswith("ecalsim.FileName"):
        line = "_".join([line, suffix])
      outf.write(line+"\n")
  return tmpf

def cgGeometry(depletion=0.01, bbthickness=0.005):
  """
  :param depletion: float, depletion depth in cm, default = 100 micrometers
  :param bbthickness: float, busbar (copper) thickness in cm, default = 50 micrometers
  """
  with open("GeometricVariables.geo", "w") as f:
    f.write(f"Constant DepletionDepth {depletion}\n")
    f.write(f"Constant BBThickness {bbthickness}\n")

if __name__ == "__main__":
  parser = argparse.ArgumentParser("")
  parser.add_argument("-s", "--sourcefile", required=True, help="Path to source file to run multiple times")
  args = parser.parse_args()
  for depth in (np.arange(2)+1)*0.001: # 10 micrometers to 200 micrometers
    cgGeometry(depth)
    tmpsrc = cgSimName(args.sourcefile, f"d{depth}")
    subprocess.run(["cosima", tmpsrc])








#
#def copywc(source, dest):
#  for f in glob.glob(source):
#    shutil.copy(f"{f}", dest)
#
#if __name__ == "__main__":
#  # Gather source to run
#  parser = argparse.ArgumentParser("")
#  parser.add_argument("-s", "--sourcefile", required=True, help="Path to source file to run multiple times")
#  args = parser.parse_args()
#
#  # Prepare simulations
#  sourcefile = args.sourcefile.split("/")[-1]
#  datafolder = f"simulations/{sourcefile[:-7]}"
#  pathlib.Path(datafolder).mkdir(exist_ok=True)
#  shutil.copy(f"simulations/{sourcefile}", f"{datafolder}/{sourcefile}")
#  # Backup geometry files
#  os.system("mkdir -p geometrybackup; rm geometrybackup/*")
#  copywc(r"simulations/*SiPixel*", "geometrybackup/")
#
#  
#
#def dontrun():
#  try:
#    for depthcm in (np.arange(20)+1)*0.001: # 10 micrometers to 200 micrometers
#      os.system()
#      subprocess.run(["cosima", "$PWD/simulations/Cs-137.source"])
#  except Exception as e:
#    copywc(r"geometrybackup/*", ".")
#    raise e
#  copywc(r"geometrybackup/*", ".")
#
#
#
#
