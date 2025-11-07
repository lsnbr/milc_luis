#!/bin/bash


#SBATCH -J flow_meas_branch0cc
#SBATCH --mail-type=END
#SBATCH -e /work/scratch/ln29bamu/outputs/%x.err.%j
#SBATCH -o /work/scratch/ln29bamu/outputs/%x.out.%j
#
#SBATCH -N 1                 # Anzahl der Nodes
#SBATCH --exclusive
#SBATCH -n 64                # Anzahl der Prozesse
#SBATCH --mem-per-cpu=1G     # Hauptspeicher in MByte pro MPI-Task
#SBATCH -t 18:00:00                # in Stunden, Minuten und Sekunden (-t 01:30:00), oder (-t 10) - nur Minuten



# -------------------------------
# Anschließend schreiben Sie Ihre eigenen Befehle zum Start der Berechnung, wie z.B.
module purge
module load openmpi python
source ~/myenv/bin/activate

# Directory of code
cd /home/ln29bamu/code/milc_luis/zeugs

# Begin work (two arguments: first a str specifing which folder in scratch/gauge_configs, second an int specifzing the last gauge config)
python do_flow_meas.py "branch0cc" 22200


EXITCODE=$?

# Aufräum- und Kopierbefehle:

# JobScript mit dem Status des wiss. Programms beenden
exit $EXITCODE
