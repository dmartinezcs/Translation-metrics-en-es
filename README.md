# Translation metrics
For now the code of each of the metrics is executed in each own .py
All of the .py write in the same .csv

## Candidates
Google Translate

## Bert
Working properly (May take a while)

## BLEU
Working properly

## Comet
Working properly (RAM amount is too small in codespaces, therefore using lowest ram consuming model)


## Meteor
Working properly

## Texts
The text shall be three corpus as to try all possibilities in the written language<br/>
A coloquial/simple one<br/>
A technical/engineering one<br/>
A philosphical/emotional one

## How to make the project work

/scripts level<br/>
python -m pip install nltk bert-score unbabel-comet torch transformers sentencepiece<br/>
python evaluate_[metric].py <br/>
python ./evaluate_[metric].py <br/>

Process may take a while depending on metric (comet may require various executions)<br/>
results appear in evaluation.csv
