# Copy the folder named rlbench.zip from /project_data/held/baeisner/rlbench.zip to /scratch/baeisner/data/rlbench.zip
# Unzip the file rlbench.zip in /scratch/baeisner/data/rlbench
# Delete the file rlbench.zip in /scratch/baeisner/data/rlbench.zip

# Rsync with progress
rsync -av --progress /project_data/held/baeisner/rlbench.zip /scratch/baeisner/data/rlbench.zip

# Unzip
unzip /scratch/baeisner/data/rlbench.zip -d /scratch/baeisner/data/rlbench

# Delete
rm /scratch/baeisner/data/rlbench.zip
