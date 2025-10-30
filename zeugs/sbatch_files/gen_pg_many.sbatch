#!/bin/bash


#SBATCH -J pg_gen_branchc
#SBATCH --mail-type=END
#SBATCH -e /work/scratch/ln29bamu/outputs/branch1-15c_runs/%x.err.%j
#SBATCH -o /work/scratch/ln29bamu/outputs/branch1-15c_runs/%x.out.%j
#
#SBATCH -N 1                 # Anzahl der Nodes
#SBATCH -n 96                # Anzahl der Prozesse
#SBATCH --mem-per-cpu=1G     # Hauptspeicher in MByte pro MPI-Task
#SBATCH -t 1-                # in Stunden, Minuten und Sekunden (-t 01:30:00), oder (-t 10) - nur Minuten



# -------------------------------
# Anschließend schreiben Sie Ihre eigenen Befehle zum Start der Berechnung, wie z.B.
module purge
module load openmpi python

# Directory of code
cd /home/ln29bamu/code/milc_luis/zeugs

# Directory for logs
OUTDIR=/work/scratch/ln29bamu/outputs/branch1-14c_runs


TOTAL_CORES=$SLURM_NTASKS
NRUNS=3
CORES_PER_RUN=$(( TOTAL_CORES / NRUNS ))


# Parameters for each run
# !!!!! important
# determine into which folder configs are saved
param_set1=("1" "2" "3")    # branch numbers


for i in $(seq 0 $((NRUNS-1))); do
    p1="${param_set1[$i]}"

    outfile="$OUTDIR/${SLURM_JOB_NAME}_branch${p1}.out.${SLURM_JOB_ID}"
    errfile="$OUTDIR/${SLURM_JOB_NAME}_branch${p1}.err.${SLURM_JOB_ID}"

    echo "Launching Python job $i with $CORES_PER_RUN cores, branch $p1"
    python3 do_gen_pg_ora_many.py "$CORES_PER_RUN" "$p1" >"$outfile" 2>"$errfile" &
done


# Wait for all background jobs to finish
wait



EXITCODE=$?


# Aufräum- und Kopierbefehle:


# JobScript mit dem Status des wiss. Programms beenden
exit $EXITCODE