# fakenewschallenge
Solving problem of text classification from Fake News Challenge

1. Problem definition

This project was based on Fake News Challenge competition. The problem description is depicted by the organizers:
"The goal of the Fake News Challenge is to explore how artificial intelligence technologies, particularly machine learning and natural language processing, might be leveraged to combat the fake news problem. We believe that these AI technologies hold promise for significantly automating parts of the procedure human fact checkers use today to determine if a story is real or a hoax."

To be more specific - the task of this challenge was text classification. The text classification was based on the relation between article headline and article body. There are 4 possible classes. From the organizers website:
"Agrees: The body text agrees with the headline.
Disagrees: The body text disagrees with the headline.
Discusses: The body text discuss the same topic as the headline, but does not take a position
Unrelated: The body text discusses a different topic than the headline"

2. Data description

The data was already split on train and test sets. Moreover, the article bodies and headlines were delivered also in separate files.

In total train set consisted of 49972 observations. The class distribution in train set is as follows:
- unrelated: 36545 (73.0%)
- discuss: 8909 (18.0%)
- agree: 3678 (7.0%)
- disagree: 840 (2.0%)

The composition of test set (25413 observations) was:
- unrelated: 18349 (72.0%)
- discuss: 4464 (18.0%)
- agree: 1903 (7.0%)
- disagree: 697 (3.0%)

3. Data Processing

There were several steps of data manipulation before it was put into the model.
Firstly, all signs in headlines and article bodies apart from letters (capital and small) were removed. Secondly, all the texts were tokenized. Thirdly, all tokens were turned into lowercase. Finally, stop words were removed. Some other data manipulation techniques (such as stemming or lemmatizing) were tried, but they didn't improve the scoring.

The next steps were based on UCL's model for Fake News Challenge. For train and test sets I constructed a matrix (k_news, 2 * n_vocabulary + 1) as follows:
- I learn a vocabulary of n_vocabulary most frequent words for a list of unique headlines and bodies from train set
- For each news from train and test sets I:
   * calculated term frequency vectors TF_h and TF_b for news headline and body respectively. 
   * Similarly I calculated TF_IDF_h and TF_IDF_b vectors
   * I calculated cosine similarity value <cos_value> of TF_IDF_h and TF_IDF_b vectors
   * Finally I squeezed TF_h, TF_b, <cos_value> to the one vector of a dim 2 * n_vocabulary + 1.

I fitted TfidfVectorizer with a list of unique headlines and bodies from train set when process both: train and test sets.
Thus obtained matrix is then passed as an input to classifiers.

4. Model

8 algorithms have been tested that support multiclass classification. For each model I calculated:

- Accuracy
- F1 score weighted
- F1 score, class Agree
- F1 score, class Disagree
- F1 score, class Discuss
- F1 score, class Unrelated
- Time of fitting the model (in minutes)

The next step was running Voting Classifier in order to optimise results. For Voting Classifier I choose these models which have in general high accuracy, but also they have relatively good F1 scores on other classes than Unrelated. Therefore I choose:
- Linear SVC (42% on class Agree and 61% on Discuss)
- Logistic Regression (45% on class Agree and 67% on Discuss)
- Decision Tree and Extra Tree (on class Disagree respectively 7% and 11%)

These models have been passed also to Grid Search in order to find the best parameters.
It turned out that for:
- Logistic Regression default parameters were optimal
- Linear SVC: C=0.5
- Decision Tree and Extra Tree: criterion='entropy', min_samples_leaf=10

Finally, Voting Classifier was ran again with optimised parameters.

It turns out that the best model in terms of accuracy and F1 score weighted is Logistic Regression set for multiclass classification with default parameters.
