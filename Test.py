import nltk
from nltk.corpus import stopwords
import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier


def get_meaningful_words(raw_text):
    # remove html tag
    html_train = BeautifulSoup(raw_text,features="html.parser")
    no_tag_text = html_train.get_text()
    # find-and-replace NOT a-zA-Z
    letters_only = re.sub("[^a-zA-Z]", " ", no_tag_text)
    # all letters to lower case
    words = letters_only.lower().split()
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops]
    return " ".join(meaningful_words)
# open file,count row number
train = pd.read_csv( "labeledTrainData.tsv", header=0, delimiter="\t", quoting=3 )
row_count = train["review"].size
# update raw data
clean_train_reviews=[]
for i in range(row_count):
    clean_train_reviews.append(get_meaningful_words(train["review"][i]))
# Creating the bag of words
vectorizer = \
    CountVectorizer(analyzer = "word",tokenizer = None,preprocessor = None,stop_words = None,max_features = 5000)
train_data_features = vectorizer.fit_transform(clean_train_reviews).toarray()

# Random Forest
forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit( train_data_features, train["sentiment"] )

# read test data and predict
test = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3 )
num_reviews = len(test["review"])
clean_test_reviews = []
for i in range(num_reviews):
    clean_review = get_meaningful_words(test["review"][i])
    clean_test_reviews.append(clean_review)

test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()
# predict
result = forest.predict(test_data_features)
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3 )