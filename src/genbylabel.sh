#!/bin/sh
# help to train walking left by separately

SEQLEN=300

EXPERIMENT_FOLDER='../Experiments_1203_multilen'
RESULT_FOLDER="dsensor_${SEQLEN}"
FILE_NAME='main_sensordiff.py'
CONFIG_NAME="uschad_genbylabel_${SEQLEN}"
GPU=1

# train
python $FILE_NAME --seqlen $SEQLEN --configname $CONFIG_NAME --resultfolder $EXPERIMENT_FOLDER --foldername $RESULT_FOLDER --gpu $GPU --train

# test
ACTIVITY='elevatordown'
python $FILE_NAME --seqlen $SEQLEN --configname $CONFIG_NAME --resultfolder $EXPERIMENT_FOLDER --foldername $RESULT_FOLDER --activityname $ACTIVITY --gpu $GPU --testid 10

ACTIVITY='elevatorup'
python $FILE_NAME --seqlen $SEQLEN --configname $CONFIG_NAME --resultfolder $EXPERIMENT_FOLDER --foldername $RESULT_FOLDER --activityname $ACTIVITY --gpu $GPU --testid 10

ACTIVITY='jumping'
python $FILE_NAME --seqlen $SEQLEN --configname $CONFIG_NAME --resultfolder $EXPERIMENT_FOLDER --foldername $RESULT_FOLDER --activityname $ACTIVITY --gpu $GPU --testid 10

ACTIVITY='runningforward'
python $FILE_NAME --seqlen $SEQLEN --configname $CONFIG_NAME --resultfolder $EXPERIMENT_FOLDER --foldername $RESULT_FOLDER --activityname $ACTIVITY --gpu $GPU --testid 10

ACTIVITY='sitting'
python $FILE_NAME --seqlen $SEQLEN --configname $CONFIG_NAME --resultfolder $EXPERIMENT_FOLDER --foldername $RESULT_FOLDER --activityname $ACTIVITY --gpu $GPU --testid 10

ACTIVITY='sleeping'
python $FILE_NAME --seqlen $SEQLEN --configname $CONFIG_NAME --resultfolder $EXPERIMENT_FOLDER --foldername $RESULT_FOLDER --activityname $ACTIVITY --gpu $GPU --testid 10

ACTIVITY='standing'
python $FILE_NAME --seqlen $SEQLEN --configname $CONFIG_NAME --resultfolder $EXPERIMENT_FOLDER --foldername $RESULT_FOLDER --activityname $ACTIVITY --gpu $GPU --testid 10

ACTIVITY='walkingdownstairs'
python $FILE_NAME --seqlen $SEQLEN --configname $CONFIG_NAME --resultfolder $EXPERIMENT_FOLDER --foldername $RESULT_FOLDER --activityname $ACTIVITY --gpu $GPU --testid 10

ACTIVITY='walkingforward'
python $FILE_NAME --seqlen $SEQLEN --configname $CONFIG_NAME --resultfolder $EXPERIMENT_FOLDER --foldername $RESULT_FOLDER --activityname $ACTIVITY --gpu $GPU --testid 10

ACTIVITY='walkingleft'
python $FILE_NAME --seqlen $SEQLEN --configname $CONFIG_NAME --resultfolder $EXPERIMENT_FOLDER --foldername $RESULT_FOLDER --activityname $ACTIVITY --gpu $GPU --testid 10

ACTIVITY='walkingright'
python $FILE_NAME --seqlen $SEQLEN --configname $CONFIG_NAME --resultfolder $EXPERIMENT_FOLDER --foldername $RESULT_FOLDER --activityname $ACTIVITY --gpu $GPU --testid 10

ACTIVITY='walkingupstairs'
python $FILE_NAME --seqlen $SEQLEN --configname $CONFIG_NAME --resultfolder $EXPERIMENT_FOLDER --foldername $RESULT_FOLDER --activityname $ACTIVITY --gpu $GPU --testid 10
