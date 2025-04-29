# Amazon Fine Food Reviews Sentiment Analysis

This repository is solution to a task given to machine learning intern candidates at Seznam.cz. 

## Task Description 
Úloha č. 3 - analýza sentimentu recenze Každá recenze má u sebe číselné hodnocení produktu na škále 1-5, titulek recenze a textový obsah recenze. Na základě textových polí titulku nebo/a obsahu predikujte číselné hodnocení produktu. Jako míru kvality predikce můžete použít např. RMSE. Pro ukázku řešení si připravte několik krátkých textů a ukázek predikcí, aby bylo vidět, podle čeho se model rozhoduje.

## Data
Data is freely available at: [https://snap.stanford.edu/data/web-FineFoods.html](https://snap.stanford.edu/data/web-FineFoods.html)

Data augmentation was run on dataset using DataAugmenter defined in data/data_augmentation.py

For data download, run: 
~~~bash
python sentiment-analysis-amazon/data/download_data.py --url http://snap.stanford.edu/data/finefoods.txt.gz
~~~