#!/bin/sh
# help to train walking left by separately


"""
Within Diffusion, the model structure changed into context type, so can't run the normal train procedure, 
after the genbylabel is done, changed it back and run these several lines
"""
# ACTIVITY='walkingleft'
# python main_activity_gen_fullpic.py --configname uschad_generate_activity --foldername $ACTIVITY --activityname $ACTIVITY --train
# python main_activity_gen_fullpic.py --configname uschad_generate_activity --foldername $ACTIVITY --activityname $ACTIVITY --testid 10


# train
python main_genbylabel.py --configname uschad_genbylabel --foldername genbylabel --gpu 6 --train

# test
ACTIVITY='elevatordown'
python main_genbylabel.py --configname uschad_genbylabel --foldername genbylabel --activityname $ACTIVITY --gpu 6 --testid 10

ACTIVITY='elevatorup'
python main_genbylabel.py --configname uschad_genbylabel --foldername genbylabel --activityname $ACTIVITY --gpu 6 --testid 10

ACTIVITY='jumping'
python main_genbylabel.py --configname uschad_genbylabel --foldername genbylabel --activityname $ACTIVITY --gpu 6 --testid 10

ACTIVITY='runningforward'
python main_genbylabel.py --configname uschad_genbylabel --foldername genbylabel --activityname $ACTIVITY --gpu 6 --testid 10

ACTIVITY='sitting'
python main_genbylabel.py --configname uschad_genbylabel --foldername genbylabel --activityname $ACTIVITY --gpu 6 --testid 10

ACTIVITY='sleeping'
python main_genbylabel.py --configname uschad_genbylabel --foldername genbylabel --activityname $ACTIVITY --gpu 6 --testid 10

ACTIVITY='standing'
python main_genbylabel.py --configname uschad_genbylabel --foldername genbylabel --activityname $ACTIVITY --gpu 6 --testid 10

ACTIVITY='walkingleft'
python main_genbylabel.py --configname uschad_genbylabel --foldername genbylabel --activityname $ACTIVITY --gpu 6 --testid 10

ACTIVITY='walkingdownstairs'
python main_genbylabel.py --configname uschad_genbylabel --foldername genbylabel --activityname $ACTIVITY --gpu 6 --testid 10

ACTIVITY='walkingforward'
python main_genbylabel.py --configname uschad_genbylabel --foldername genbylabel --activityname $ACTIVITY --gpu 6 --testid 10

ACTIVITY='walkingleft'
python main_genbylabel.py --configname uschad_genbylabel --foldername genbylabel --activityname $ACTIVITY --gpu 6 --testid 10

ACTIVITY='walkingright'
python main_genbylabel.py --configname uschad_genbylabel --foldername genbylabel --activityname $ACTIVITY --gpu 6 --testid 10

ACTIVITY='walkingupstairs'
python main_genbylabel.py --configname uschad_genbylabel --foldername genbylabel --activityname $ACTIVITY --gpu 6 --testid 10
