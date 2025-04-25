#!/bin/bash
#SBATCH -J mtl                                  # Job name
#SBATCH -A gts-dmahajan7                 # charge account
#SBATCH -N1                                     # Number of nodes and cores per node required
#SBATCH --mem-per-gpu=64G                       # Memory per core
#SBATCH -t 2:00:00                              # Duration of the job (Ex: 15 mins)
#SBATCH -q embers                              # QOS Name
#SBATCH --gres=gpu:h100:1
#SBATCH -o Report-%j.out                        # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=slee3794@gatech.edu        # E-mail address for notifications
cd $SLURM_SUBMIT_DIR                            # Change to working directory
# srun python main.py full=full model.backbone=b0 model=efficientnet dataset=eurosat
# srun python main.py full=full model.backbone=b1 model=efficientnet dataset=eurosat
# srun python main.py full=full model.backbone=b2 model=efficientnet dataset=eurosat
# srun python main.py full=full model.backbone=b3 model=efficientnet dataset=eurosat
# srun python main.py full=full model.backbone=b4 model=efficientnet dataset=eurosat
# srun python main.py full=full model.backbone=b5 model=efficientnet dataset=eurosat
srun python main.py full=full model.backbone=b6 model=efficientnet dataset=eurosat
srun python main.py full=full model.backbone=b7 model=efficientnet dataset=eurosat
srun python main.py full=head model.backbone=b0 model=efficientnet dataset=eurosat
srun python main.py full=head model.backbone=b1 model=efficientnet dataset=eurosat
srun python main.py full=head model.backbone=b2 model=efficientnet dataset=eurosat
srun python main.py full=head model.backbone=b3 model=efficientnet dataset=eurosat
srun python main.py full=head model.backbone=b4 model=efficientnet dataset=eurosat
srun python main.py full=head model.backbone=b5 model=efficientnet dataset=eurosat
srun python main.py full=head model.backbone=b6 model=efficientnet dataset=eurosat
srun python main.py full=head model.backbone=b7 model=efficientnet dataset=eurosat

# srun python main.py full=full model.backbone=b0 model=efficientnet dataset=
# srun python main.py full=full model.backbone=b1 model=efficientnet dataset=
# srun python main.py full=full model.backbone=b2 model=efficientnet dataset=
# srun python main.py full=full model.backbone=b3 model=efficientnet dataset=
# srun python main.py full=full model.backbone=b4 model=efficientnet dataset=
# srun python main.py full=full model.backbone=b5 model=efficientnet dataset=
# srun python main.py full=full model.backbone=b6 model=efficientnet dataset=
# srun python main.py full=full model.backbone=b7 model=efficientnet dataset=
# srun python main.py full=half model.backbone=b0 model=efficientnet dataset=
# srun python main.py full=half model.backbone=b1 model=efficientnet dataset=
# srun python main.py full=half model.backbone=b2 model=efficientnet dataset=
# srun python main.py full=half model.backbone=b3 model=efficientnet dataset=
# srun python main.py full=half model.backbone=b4 model=efficientnet dataset=
# srun python main.py full=half model.backbone=b5 model=efficientnet dataset=
# srun python main.py full=half model.backbone=b6 model=efficientnet dataset=
# srun python main.py full=half model.backbone=b7 model=efficientnet dataset=

# srun python main.py full=full model.backbone=b0 model=deeplabv3 dataset=landcover_ai
# srun python main.py full=full model.backbone=b1 model=deeplabv3 dataset=landcover_ai
# srun python main.py full=full model.backbone=b2 model=deeplabv3 dataset=landcover_ai
# srun python main.py full=full model.backbone=b3 model=deeplabv3 dataset=landcover_ai
# srun python main.py full=full model.backbone=b4 model=deeplabv3 dataset=landcover_ai
# srun python main.py full=full model.backbone=b5 model=deeplabv3 dataset=landcover_ai
# srun python main.py full=full model.backbone=b6 model=deeplabv3 dataset=landcover_ai
# srun python main.py full=full model.backbone=b7 model=deeplabv3 dataset=landcover_ai
# srun python main.py full=half model.backbone=b0 model=deeplabv3 dataset=landcover_ai
# srun python main.py full=half model.backbone=b1 model=deeplabv3 dataset=landcover_ai
# srun python main.py full=half model.backbone=b2 model=deeplabv3 dataset=landcover_ai
# srun python main.py full=half model.backbone=b3 model=deeplabv3 dataset=landcover_ai
# srun python main.py full=half model.backbone=b4 model=deeplabv3 dataset=landcover_ai
# srun python main.py full=half model.backbone=b5 model=deeplabv3 dataset=landcover_ai
# srun python main.py full=half model.backbone=b6 model=deeplabv3 dataset=landcover_ai
# srun python main.py full=half model.backbone=b7 model=deeplabv3 dataset=landcover_ai

# srun python main.py full=full model.backbone=b0 model=deeplabv3 dataset=
# srun python main.py full=full model.backbone=b1 model=deeplabv3 dataset=
# srun python main.py full=full model.backbone=b2 model=deeplabv3 dataset=
# srun python main.py full=full model.backbone=b3 model=deeplabv3 dataset=
# srun python main.py full=full model.backbone=b4 model=deeplabv3 dataset=
# srun python main.py full=full model.backbone=b5 model=deeplabv3 dataset=
# srun python main.py full=full model.backbone=b6 model=deeplabv3 dataset=
# srun python main.py full=full model.backbone=b7 model=deeplabv3 dataset=
# srun python main.py full=half model.backbone=b0 model=deeplabv3 dataset=
# srun python main.py full=half model.backbone=b1 model=deeplabv3 dataset=
# srun python main.py full=half model.backbone=b2 model=deeplabv3 dataset=
# srun python main.py full=half model.backbone=b3 model=deeplabv3 dataset=
# srun python main.py full=half model.backbone=b4 model=deeplabv3 dataset=
# srun python main.py full=half model.backbone=b5 model=deeplabv3 dataset=
# srun python main.py full=half model.backbone=b6 model=deeplabv3 dataset=
# srun python main.py full=half model.backbone=b7 model=deeplabv3 dataset=