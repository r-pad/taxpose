
# Rsync with progress
rsync -av --progress /project_data/held/baeisner/rlbench10_collisions_val.tar.gz /scratch/baeisner/data/rlbench_val.tar.gz
rsync -av --progress /project_data/held/baeisner/rlbench10_collisions.tar.gz /scratch/baeisner/data/rlbench10_collisions.tar.gz

# Extract
tar -xvzf /scratch/baeisner/data/rlbench_val.tar.gz -C /scratch/baeisner/data/
tar -xvzf /scratch/baeisner/data/rlbench10_collisions.tar.gz -C /scratch/baeisner/data/

# Delete
rm /scratch/baeisner/data/rlbench_val.tar.gz
rm /scratch/baeisner/data/rlbench10_collisions.tar.gz
