import os, sys, inspect

#################################
#    Define parent directory    #
#################################

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
potatonewdir = os.path.dirname(currentdir)
sys.path.insert(0, potatonewdir)

vaedir = os.path.dirname(potatonewdir)
#sys.path.insert(0, vaedir)

parentdir = os.path.dirname(vaedir)
sys.path.insert(0, parentdir)

FIGURES_FOLDER       = f'{currentdir}/SavedFigures'
MEASUREMENTS_FOLDER  = f'{currentdir}/SavedMeasurements'
