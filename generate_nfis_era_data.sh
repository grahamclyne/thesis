
#!/bin/bash


if [ $HOSTNAME == "gpe-er14317-06m.concordia.ca" ];
then
    export NUM_CORES=`sysctl -n hw.ncpu`
    export PROJECT_PATH='/Users/gclyne/thesis'
    export NFIS_PATH='/Users/gclyne/thesis/data/NFIS'
elif [ $HOSTNAME == "cedar5.cedar.computecanada.ca" ];
then 
    #SBATCH --time=12:00:00
    #SBATCH --account=def-dmatthew
    #SBATCH --cpus-per-task=32
    module load python/3.9
    export NFIS_PATH='~/scratch'
    export NUM_CORES=32
    export PROJECT_PATH='/home/gclyne/projects/def-dmatthew/gclyne/thesis'
    source ~/ENV/bin/activate
fi


python ${PROJECT_PATH}/other/generate_nfis_era_data.py