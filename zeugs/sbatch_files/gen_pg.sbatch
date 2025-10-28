#!/bin/bash

#SBATCH -J pg_gen
#SBATCH --mail-type=END
#SBATCH -e /work/scratch/ln29bamu/outputs/%x.err.%j
#SBATCH -o /work/scratch/ln29bamu/outputs/%x.out.%j
#
#SBATCH -n 64                # Anzahl der Prozesse
#SBATCH --mem-per-cpu=500    # Hauptspeicher in MByte pro MPI-Task
#SBATCH -t 5                 # in Stunden, Minuten und Sekunden (-t 01:30:00), oder (-t 10) - nur Minuten

# -------------------------------
# Anschließend schreiben Sie Ihre eigenen Befehle zum Start der Berechnung, wie z.B.
module purge
module load openmpi python py-numpy
cd /home/ln29bamu/code/milc_luis/zeugs
python3 do_gen_pg_ora.py
EXITCODE=$?

# Aufräum- und Kopierbefehle:
...
# JobScript mit dem Status des wiss. Programms beenden
exit $EXITCODE