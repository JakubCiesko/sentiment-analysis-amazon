# Amazon Fine Food Reviews Sentiment Analysis

This repository is solution to a task given to machine learning intern candidates at Seznam.cz. 

## Task Description 
Úloha č. 3 - analýza sentimentu recenze Každá recenze má u sebe číselné hodnocení produktu na škále 1-5, titulek recenze a textový obsah recenze. Na základě textových polí titulku nebo/a obsahu predikujte číselné hodnocení produktu. Jako míru kvality predikce můžete použít např. RMSE. Pro ukázku řešení si připravte několik krátkých textů a ukázek predikcí, aby bylo vidět, podle čeho se model rozhoduje.

## Data
Data is freely available at: [https://snap.stanford.edu/data/web-FineFoods.html](https://snap.stanford.edu/data/web-FineFoods.html)

Data augmentation was run on dataset using DataAugmenter defined in data/data_augmentation.py with q = 0.75
Augmented data is freely available here: [https://huggingface.co/datasets/maennyn/amazon_fine_foods_augmented_q75](https://huggingface.co/datasets/maennyn/amazon_fine_foods_augmented_q75)


For base data download, run: 
~~~bash
python sentiment-analysis-amazon/data/download_data.py --url http://snap.stanford.edu/data/finefoods.txt.gz
~~~

## Used Methods and models

As basic ML baselines, Naive Bayes and Logistic Regression were used. The code is available in baselines.ipynb. The text features (text and summary) were first concatenated, and then tf-idf score was computed which was the input for Naive Bayes and Logistic Regression. These models showed good results (available in metrics, reports directory, or in the mentioned notebook), and are highly interpretable which is a huge advantage.
The final model is a finetuned version of base RoBERTa. There are two versions available here: Better one, trained for 6 epochs: [https://huggingface.co/maennyn/roberta-amazon-finefood-sentiment6e](https://huggingface.co/maennyn/roberta-amazon-finefood-sentiment6e), a slightly worse one trained for 3 epochs: [https://huggingface.co/maennyn/roberta-amazon-finefood-sentiment](https://huggingface.co/maennyn/roberta-amazon-finefood-sentiment). The worse one was trained on the original dataset which was balanced through downsampling. The better one was trained on the augmented dataset which was also balanced through downsampling. The training of better model was stopped due to computational reasons. The model could be, I believe, improved with a longer training and hyperparameter tuning using for example optuna.
Final best results are: accuracy of 0.80, macro F1-score of 0.80, and RMSE of 0.54 on the test set. Precision and recall scores are well-balanced across all sentiment classes, with particularly strong performance on the extreme ends (0 and 4).