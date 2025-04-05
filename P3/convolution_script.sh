#!/bin/bash
#----------------------------------------------------
# Example SLURM job script with SBATCH
#----------------------------------------------------
#SBATCH -J myjob            # Job name
#SBATCH -o myjob_%j.o       # Name of stdout output file(%j expands to jobId)
#SBATCH -e myjob_%j.e       # Name of stderr output file(%j expands to jobId)
#SBATCH -c 8                # Cores per task requested
#SBATCH -t 00:10:00         # Run time (hh:mm:ss) - 10 min
#SBATCH --mem-per-cpu=3G    # Memory per core demandes (24 GB = 3GB * 8 cores)

image=grayscale_by_rows.pgm
hilos=(2 4 8 16 32 48 64)
chunks=(2 4 8 16 32 48 64)

for hilo in "${hilos[@]}"
do
    for chunk in "${chunks[@]}"
    do
        echo "H: $hilo  C: $chunk"
        ./$1 $image $hilo $chunk
    done
done

