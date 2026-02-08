#!/bin/bash


#SBATCH -J pg_gen_0p99Tc_b20
#SBATCH --mail-type=END
#SBATCH -e /work/scratch/ln29bamu/outputs/gen_0p99Tc_branches/%x.err.%j
#SBATCH -o /work/scratch/ln29bamu/outputs/gen_0p99Tc_branches/%x.out.%j
#
#SBATCH -N 1                 # Anzahl der Nodes
#SBATCH --exclusive
#SBATCH -n 96                # Anzahl der Prozesse
#SBATCH --mem-per-cpu=1G     # Hauptspeicher in MByte pro MPI-Task
#SBATCH -t 1-                # in Stunden, Minuten und Sekunden (-t 01:30:00), oder (-t 10) - nur Minuten, oder (-t 1-) ist ein Tag



# -------------------------------
# Anschließend schreiben Sie Ihre eigenen Befehle zum Start der Berechnung, wie z.B.
module purge
module load openmpi python
source ~/myenv/bin/activate

# Directory of code
cd /home/ln29bamu/code/milc_luis/zeugs

# Begin work
python3 do_gen_pg_ora.py 20


EXITCODE=$?

# Aufräum- und Kopierbefehle:

# JobScript mit dem Status des wiss. Programms beenden
exit $EXITCODE