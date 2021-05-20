import os, sys, inspect

#################################
#    Define parent directory    #
#################################

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
vaedir = os.path.dirname(currentdir)
sys.path.insert(0, vaedir)

parentdir = os.path.dirname(vaedir)
sys.path.insert(0, parentdir)

#evaldir = f'{currentdir}/evaluations'
#sys.path.insert(0, evaldir)

##################################################
# Define work direction for generated datasets   #
##################################################

PATH = f'{parentdir}/generated_dataset/Potato/'

#PERFECT_AUG  = f'{PATH}/NormalAugmentations'
#METAL_AUG    = f'{PATH}/Anomaly1Augmentations'
#HOLLOW_AUG   = f'{PATH}/Anomaly2Augmentations'

DATA_AUGMENTATION = f'{PATH}/DataAugmentation'
NORMAL_TRAIN_AUG  = f'{DATA_AUGMENTATION}/NormalTrainAugmentations'
NORMAL_VAL_AUG    = f'{DATA_AUGMENTATION}/NormalValidationAugmentations'
NORMAL_TEST_AUG   = f'{DATA_AUGMENTATION}/NormalTestAugmentations'
ANOMALY1_AUG      = f'{DATA_AUGMENTATION}/Anomaly1Augmentations'
ANOMALY2_AUG      = f'{DATA_AUGMENTATION}/Anomaly2Augmentations'


#################################
#    Define image dimensions    #
#################################
hh, ww = 128, 128