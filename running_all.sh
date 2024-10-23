#!/bin/sh

ACTIVITY='elevatordown'
# python main_activity_gen_fullpic.py --configname uschad_generate_activity --foldername $ACTIVITY --activityname $ACTIVITY --train
python main_activity_gen_fullpic.py --configname uschad_generate_activity --foldername $ACTIVITY --activityname $ACTIVITY --testid 10

ACTIVITY='elevatorup'
# python main_activity_gen_fullpic.py --configname uschad_generate_activity --foldername $ACTIVITY --activityname $ACTIVITY --train
python main_activity_gen_fullpic.py --configname uschad_generate_activity --foldername $ACTIVITY --activityname $ACTIVITY --testid 10

ACTIVITY='jumping'
# python main_activity_gen_fullpic.py --configname uschad_generate_activity --foldername $ACTIVITY --activityname $ACTIVITY --train
python main_activity_gen_fullpic.py --configname uschad_generate_activity --foldername $ACTIVITY --activityname $ACTIVITY --testid 10

ACTIVITY='runningforward'
# python main_activity_gen_fullpic.py --configname uschad_generate_activity --foldername $ACTIVITY --activityname $ACTIVITY --train
python main_activity_gen_fullpic.py --configname uschad_generate_activity --foldername $ACTIVITY --activityname $ACTIVITY --testid 10

ACTIVITY='sitting'
# python main_activity_gen_fullpic.py --configname uschad_generate_activity --foldername $ACTIVITY --activityname $ACTIVITY --train
python main_activity_gen_fullpic.py --configname uschad_generate_activity --foldername $ACTIVITY --activityname $ACTIVITY --testid 10

ACTIVITY='sleeping'
# python main_activity_gen_fullpic.py --configname uschad_generate_activity --foldername $ACTIVITY --activityname $ACTIVITY --train
python main_activity_gen_fullpic.py --configname uschad_generate_activity --foldername $ACTIVITY --activityname $ACTIVITY --testid 10

ACTIVITY='standing'
# python main_activity_gen_fullpic.py --configname uschad_generate_activity --foldername $ACTIVITY --activityname $ACTIVITY --train
python main_activity_gen_fullpic.py --configname uschad_generate_activity --foldername $ACTIVITY --activityname $ACTIVITY --testid 10

ACTIVITY='walkingdownstairs'
# python main_activity_gen_fullpic.py --configname uschad_generate_activity --foldername $ACTIVITY --activityname $ACTIVITY --train
python main_activity_gen_fullpic.py --configname uschad_generate_activity --foldername $ACTIVITY --activityname $ACTIVITY --testid 10

ACTIVITY='walkingforward'
# python main_activity_gen_fullpic.py --configname uschad_generate_activity --foldername $ACTIVITY --activityname $ACTIVITY --train
python main_activity_gen_fullpic.py --configname uschad_generate_activity --foldername $ACTIVITY --activityname $ACTIVITY --testid 10

ACTIVITY='walkingleft'
# python main_activity_gen_fullpic.py --configname uschad_generate_activity --foldername $ACTIVITY --activityname $ACTIVITY --train
python main_activity_gen_fullpic.py --configname uschad_generate_activity --foldername $ACTIVITY --activityname $ACTIVITY --testid 10

ACTIVITY='walkingright'
# python main_activity_gen_fullpic.py --configname uschad_generate_activity --foldername $ACTIVITY --activityname $ACTIVITY --train
python main_activity_gen_fullpic.py --configname uschad_generate_activity --foldername $ACTIVITY --activityname $ACTIVITY --testid 10

ACTIVITY='walkingupstairs'
# python main_activity_gen_fullpic.py --configname uschad_generate_activity --foldername $ACTIVITY --activityname $ACTIVITY --train
python main_activity_gen_fullpic.py --configname uschad_generate_activity --foldername $ACTIVITY --activityname $ACTIVITY --testid 10
