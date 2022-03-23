#!/bin/bash 

module purge
module load gcc mvapich2

#NAMD=$HOME/bin/namd2-2.13-gcc-openmpi
NAMD=/projects/jdb252_1/tj227/bin/namd2-2.14-gcc-mvapich2
NEXT_DEPENDENCY=
FIRST=0
LAST=40
CHUNK_SIZE=6
JOB_NAME=CEA
PREFIX=POCE_A

# Iterate over chunks
for chunk_start in `seq $FIRST $CHUNK_SIZE $LAST`; do

chunk_end=$(($chunk_start + $CHUNK_SIZE - 1))
if (( $chunk_end > $LAST )); then
    chunk_end=$LAST
fi

SCRIPT() { cat <<RAR
#!/bin/bash
#SBATCH -J ${JOB_NAME}_${chunk_start}_${chunk_end}
#SBATCH -o out%j.amarel.log 
##SBATCH --partition=cmain --constraint=oarc -t 72:00:00
##SBATCH --partition p_jdb252_1
#SBATCH --partition p_ccib_1 --exclude=halc068,memc001
#SBATCH -t 18:00:00 
#SBATCH -N 1 -n 32 --requeue
#SBATCH --export=ALL

module purge
module load gcc mvapich2

echo "PWD: \$PWD"
echo "Modules:"
module list
echo "SLURM_NTASKS: \$SLURM_NTASKS"
echo "Nodes: \$SLURM_JOB_NODELIST"

set -e 
THEN=\`date +%s\`
for m in \`seq $chunk_start $chunk_end\`; do
m=\`printf %03d \$m\`
namdconf=\$(python3 /home/tj227/mmdevel/relentless_fep.py config_${PREFIX}.yaml \${m})
if [ -z \$namdconf ]; then continue; fi
namdlog=\$(basename \$namdconf .namd).log
srun --mpi=pmi2 $NAMD \${namdconf} > \${namdlog}
NOW=\`date +%s\`
done

DURATION_MINS=\$(((\$NOW - \$THEN) / 60))
echo "${JOB_NAME}: Done with ${PREFIX} ${chunk_start}-${chunk_end} in $PWD (Wall clock: \$DURATION_MINS min)." | ~/bin/slack-ttjoseph
RAR
}

# Show the script
SCRIPT

# Submit the job and extract the jobid
# Use the jobid to construct a dependency specification for the next job submission
JOBID=`SCRIPT | sbatch $NEXT_DEPENDENCY | tail -n1 | awk '{print $4}'`
echo "Submitted job using dependency argument: $NEXT_DEPENDENCY"

NEXT_DEPENDENCY="-d afterok:$JOBID"
# Intentionally disable the job dependencies
NEXT_DEPENDENCY=
done

