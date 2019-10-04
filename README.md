# jigsaw
Insight Data Science project, improving the search for jigsaw puzzles through puzzle image and amazon review data

This repo has 3 main parts:
1. A web app running on ElasticBeanstalk, located in the mypuzzle directory.
2. One jupyter notebook to train a model and classify puzzle images by their art style, and a second notebook to 
repeat the same procedure to classify their theme. These are located in the image_classification directory.
3. A jupyter notebook to parse amazon reviews and calculate average sentiment in a number of different categories, 
located in the review_scores directory.

The results of each model are stored in pandas dataframes, and then combined into the final clean_product_df.csv 
file used by the web app.

## Image classification
The ResNet-50 model is used for each classification. The last layer of the model was replaced and the new model 
trained on a few hundred hand-labelled images using the Adam optimizer and the negative log likelihood loss function.

## Review classification
The sentences are processed using TF-IDF to generate a feature vector, and then a random forest model is used to 
classify them by topic. The RF model was trained on 500 hand-labelled sentences.
