#!/bin/sh
# help to train walking left by separately

RESULT_FOLDER='genbylabel_1118baseline_retrain'
FILE_NAME='main_genbylabel_200.py'
CONFIG_NAME='uschad_genbylabel_200'
GPU=6

# train
# python $FILE_NAME --configname $CONFIG_NAME --foldername $RESULT_FOLDER --gpu $GPU --train

# test
ACTIVITY='elevatordown'
python $FILE_NAME --configname $CONFIG_NAME --foldername $RESULT_FOLDER --activityname $ACTIVITY --gpu $GPU --testid 10

ACTIVITY='elevatorup'
python $FILE_NAME --configname $CONFIG_NAME --foldername $RESULT_FOLDER --activityname $ACTIVITY --gpu $GPU --testid 10

ACTIVITY='jumping'
python $FILE_NAME --configname $CONFIG_NAME --foldername $RESULT_FOLDER --activityname $ACTIVITY --gpu $GPU --testid 10

ACTIVITY='runningforward'
python $FILE_NAME --configname $CONFIG_NAME --foldername $RESULT_FOLDER --activityname $ACTIVITY --gpu $GPU --testid 10

ACTIVITY='sitting'
python $FILE_NAME --configname $CONFIG_NAME --foldername $RESULT_FOLDER --activityname $ACTIVITY --gpu $GPU --testid 10

ACTIVITY='sleeping'
python $FILE_NAME --configname $CONFIG_NAME --foldername $RESULT_FOLDER --activityname $ACTIVITY --gpu $GPU --testid 10

ACTIVITY='standing'
python $FILE_NAME --configname $CONFIG_NAME --foldername $RESULT_FOLDER --activityname $ACTIVITY --gpu $GPU --testid 10

ACTIVITY='walkingdownstairs'
python $FILE_NAME --configname $CONFIG_NAME --foldername $RESULT_FOLDER --activityname $ACTIVITY --gpu $GPU --testid 10

ACTIVITY='walkingforward'
python $FILE_NAME --configname $CONFIG_NAME --foldername $RESULT_FOLDER --activityname $ACTIVITY --gpu $GPU --testid 10

ACTIVITY='walkingleft'
python $FILE_NAME --configname $CONFIG_NAME --foldername $RESULT_FOLDER --activityname $ACTIVITY --gpu $GPU --testid 10

ACTIVITY='walkingright'
python $FILE_NAME --configname $CONFIG_NAME --foldername $RESULT_FOLDER --activityname $ACTIVITY --gpu $GPU --testid 10

ACTIVITY='walkingupstairs'
python $FILE_NAME --configname $CONFIG_NAME --foldername $RESULT_FOLDER --activityname $ACTIVITY --gpu $GPU --testid 10
