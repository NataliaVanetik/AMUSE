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


If you use this code, please cite:

Litvak M, Vanetik N. Summarization of financial reports with AMUSE. In Proceedings of the 3rd Financial Narrative Processing Workshop 2021 (pp. 31-36).

@inproceedings{litvak2021summarization,
  title={Summarization of financial reports with AMUSE},
  author={Litvak, Marina and Vanetik, Natalia},
  booktitle={Proceedings of the 3rd Financial Narrative Processing Workshop},
  pages={31--36},
  year={2021}
}
