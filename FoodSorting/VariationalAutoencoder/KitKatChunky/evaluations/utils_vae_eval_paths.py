import os, sys, inspect

#################################
#    Define parent directory    #
#################################

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
kitkatdir = os.path.dirname(currentdir)
sys.path.insert(0, kitkatdir)

vaedir = os.path.dirname(kitkatdir)
#sys.path.insert(0, vaedir)

parentdir = os.path.dirname(vaedir)
sys.path.insert(0, parentdir)

FIGURES_FOLDER       = f'{currentdir}/SavedFigures'
MEASUREMENTS_FOLDER  = f'{currentdir}/SavedMeasurements'
