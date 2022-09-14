
#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --account=def-dmatthew
#SBATCH --cpus-per-task=32
export NUM_CORES=32
export PROJECT_DIR='/home/gclyne/projects/def-dmatthew/gclyne/thesis/'

module load python/3.9

source ~/
python ${PROJECT_DIR}other/generate_observable_rows.py