import os
import subprocess
import argparse



if __name__ == "__main__":
  # Gather source to run
  parser = argparse.ArgumentParser("")
  parser.add_argument("-s", "--sourcefile", required=True, help="Path to source file to run multiple times")
  args = parser.parse_args()

  # Prepare simulations
  datafolder = f"simulations/{args.sourcefile[:-6]}"
  subprocess.run(f"mkdir -p {datafolder}")
  subprocess.run(f"cp simulations/{args.sourcefile} simulations/{datafolder}/{args.sourcefile}")
  # Backup geometry files
  os.system("mkdir -p geometrybackup; rm geometrybackup/*")
  subprocess.run(f"cp simulations/*SiPixel* geometrybackup")

def not()
  try:
    for depthcm in (np.arange(20)+1)*0.001: # 10 micrometers to 200 micrometers
      os.system()
      subprocess.run(["cosima", "$PWD/simulations/Cs-137.source"])
  except Exception as e:
    os.system("cp geometrybackup/* .")
    raise e

  os.system("cp geometrybackup/* .")




