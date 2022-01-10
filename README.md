# AMUSE
AMUSE - financial summarization

Unzip data.zip

Train new model:

 python FinAnalyze.py --task train --start 0 --count <how many files,-1 for all> --modelpath data/models/new_model.h5 --train data/train --gold data/gold

data/train = dir where the text files are
data/gold  = dir where the gold summaries are


Trains new AMUSE prediction model for given files and stores it in an .h5 file

Generate summaries with existing model:

  python FinAnalyze.py --task generate-summaries --start 0 --count <how many files,-1 for all> --modelpath data/models/new_model.h5 --test data/test/ --summarydir data/summaries
  
Also stored: 

a model trained on 3000 files named model.training.muse.3000.all.h5
