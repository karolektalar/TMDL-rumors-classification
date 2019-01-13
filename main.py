import json
import os
from datetime import datetime
import nltk
from sklearn import svm, tree, naive_bayes, neural_network, neighbors
import sklearn
import numpy as np

train_data = open("rumoureval-2019-training-data/train-key.json").read()
train_data = json.loads(train_data)
train_data = train_data['subtaskbenglish']

wrongnumber = 0
token_dict = {}
nltk.download("all")
labels_array = []
tweet_id_array = []
tweet_favorite_count_array = []
tweet_retweet_count_array = []
tweet_number_of_chars_array = []
tweet_number_of_words_array = []
user_favourites_count_array = []
user_followers_count_array = []
user_date_difference_array = []
nouns_array = []
verbs_array = []
adjectives_array = []
adverbs_array = []
interjections_array = []
pronouns_array = []
retweeted_array = []

labels_array_test = []
tweet_id_array_test = []
tweet_favorite_count_array_test = []
tweet_retweet_count_array_test = []
tweet_number_of_chars_array_test = []
tweet_number_of_words_array_test = []
user_favourites_count_array_test = []
user_followers_count_array_test = []
user_date_difference_array_test = []
nouns_array_test = []
verbs_array_test = []
adjectives_array_test = []
adverbs_array_test = []
interjections_array_test = []
pronouns_array_test = []
retweeted_array_test = []


number_true = 0
number_false = 0

my_list = os.listdir('rumoureval-2019-training-data/twitter-english')
for dirs in my_list:
    tweets = os.listdir('rumoureval-2019-training-data/twitter-english/' + dirs)
    for tweet in tweets:
        nouns = 0
        verbs = 0
        adjectives = 0
        adverbs = 0
        interjections = 0
        pronouns = 0
        tweet_id = tweet
        # print(tweet_id)
        json_data = open(
            'rumoureval-2019-training-data/twitter-english/' + dirs + '/' + tweet_id + '/source-tweet/' + tweet_id + '.json').read()
        data = json.loads(json_data)

        tweet_creation_date = data['created_at']
        tweet_creation_date = datetime.strptime(tweet_creation_date, '%a %b %d %H:%M:%S %z %Y')
        tweet_text = data['text']

        text = nltk.word_tokenize(tweet_text)
        tokens = nltk.pos_tag(text)
        # print(tokens)

        for i in range(0, len(tokens) - 1):
            if tokens[i][1][0] == 'N':
                nouns = nouns + 1
            if tokens[i][1][0] == 'V':
                verbs = verbs + 1
            if tokens[i][1][0] == 'J':
                adjectives = adjectives + 1
            if tokens[i][1][0] == 'R':
                adverbs = adverbs + 1
            if tokens[i][1][0] == 'U':
                interjections = interjections + 1
            if tokens[i][1][0] == 'R':
                pronouns = pronouns + 1

        retweeted = data['retweeted']
        tweet_favorite_count = data['favorite_count']
        tweet_retweet_count = data['retweet_count']
        tweet_number_of_chars = len(tweet_text)
        tweet_number_of_words = len(tweet_text.split())
        user_favourites_count = data['user']['favourites_count']
        user_followers_count = data['user']['followers_count']
        user_creation_date = data['user']['created_at']
        user_creation_date = datetime.strptime(user_creation_date, '%a %b %d %H:%M:%S %z %Y')
        nouns = nouns / tweet_number_of_words
        verbs = verbs / tweet_number_of_words
        adjectives = adjectives / tweet_number_of_words
        adverbs = adverbs / tweet_number_of_words
        interjections = interjections / tweet_number_of_words
        pronouns = pronouns / tweet_number_of_words

        try:
            if train_data[tweet_id] != 'unverified':
                labels_array.append(train_data[tweet_id])
                tweet_id_array.append(tweet_id)
                tweet_favorite_count_array.append(tweet_favorite_count)
                tweet_retweet_count_array.append(tweet_retweet_count)
                tweet_number_of_chars_array.append(tweet_number_of_chars)
                tweet_number_of_words_array.append(tweet_number_of_words)
                user_favourites_count_array.append(user_favourites_count)
                user_followers_count_array.append(user_followers_count)
                user_date_difference_array.append((user_creation_date - tweet_creation_date).days)
                nouns_array.append(nouns)
                verbs_array.append(verbs)
                adjectives_array.append(adjectives)
                adverbs_array.append(adverbs)
                interjections_array.append(interjections)
                pronouns_array.append(pronouns)
                retweeted_array.append(retweeted)
        except:
            wrongnumber = wrongnumber + 1

#rearange arrays to numpy
numpy_tweet_favorite_count_array = np.array(tweet_favorite_count_array)
numpy_tweet_retweet_count_array = np.array(tweet_retweet_count_array)
numpy_tweet_number_of_chars_array = np.array(tweet_number_of_chars_array)
numpy_tweet_number_of_words_array = np.array(tweet_number_of_words_array)
numpy_user_favourites_count_array = np.array(user_favourites_count_array)
numpy_user_followers_count_array = np.array(user_followers_count_array)
numpy_nouns_array = np.array(nouns_array)
numpy_verbs_array = np.array(verbs_array)
numpy_user_date_difference_array = np.array(user_date_difference_array)
numpy_adjectives_array = np.array(adjectives_array)
numpy_adverbs_array = np.array(adverbs_array)
numpy_interjections_array = np.array(interjections_array)
numpy_pronouns_array = np.array(pronouns_array)
numpy_retweeted_array = np.array(retweeted_array)

x = np.column_stack((
                     numpy_pronouns_array,
                     numpy_interjections_array,
                     numpy_adverbs_array,
                     numpy_nouns_array,
                     numpy_verbs_array,
                     numpy_adjectives_array,
                     numpy_user_favourites_count_array,
                     numpy_user_followers_count_array,
                     numpy_tweet_favorite_count_array,
                     numpy_tweet_retweet_count_array,
                     numpy_tweet_number_of_chars_array,
                     numpy_tweet_number_of_words_array,
                     numpy_user_date_difference_array,
                     numpy_retweeted_array))



svm_class = svm.SVC()
y = np.array(labels_array)
svm_class = svm_class.fit(x, y)


linear_regression_class = sklearn.linear_model\
    .LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')\
    .fit(x, y)


tree_class = tree.DecisionTreeClassifier()
tree_class = tree_class.fit(x, y)


naive_bayes_class = naive_bayes.GaussianNB()
naive_bayes_class = naive_bayes_class.fit(x, y)


nn_class = neural_network.MLPClassifier(learning_rate='adaptive')
nn_class.fit(x, y)

kn_class = neighbors.KNeighborsClassifier(n_neighbors=4)
kn_class.fit(x, y)

################


train_data = open("rumoureval-2019-training-data/dev-key.json").read()
train_data = json.loads(train_data)
train_data = train_data['subtaskbenglish']




number_true = 0
number_false = 0

# os.walk("rumoureval-2019-training-data/twitter-english")
my_list = os.listdir('rumoureval-2019-training-data/twitter-english')
for dirs in my_list:
    tweets = os.listdir('rumoureval-2019-training-data/twitter-english/' + dirs)
    for tweet in tweets:
        nouns = 0
        verbs = 0
        adjectives = 0
        adverbs = 0
        interjections = 0
        pronouns = 0
        tweet_id = tweet
        json_data = open(
            'rumoureval-2019-training-data/twitter-english/' + dirs + '/' + tweet_id + '/source-tweet/' + tweet_id + '.json').read()
        data = json.loads(json_data)

        tweet_creation_date = data['created_at']
        tweet_creation_date = datetime.strptime(tweet_creation_date, '%a %b %d %H:%M:%S %z %Y')
        tweet_text = data['text']

        text = nltk.word_tokenize(tweet_text)
        tokens = nltk.pos_tag(text)

        for i in range(0, len(tokens) - 1):
            if tokens[i][1][0] == 'N':
                nouns = nouns + 1
            if tokens[i][1][0] == 'V':
                verbs = verbs + 1
            if tokens[i][1][0] == 'J':
                adjectives = adjectives + 1
            if tokens[i][1][0] == 'R':
                adverbs = adverbs + 1
            if tokens[i][1][0] == 'U':
                interjections = interjections + 1
            if tokens[i][1][0] == 'R':
                pronouns = pronouns + 1

        retweeted = data['retweeted']
        tweet_favorite_count = data['favorite_count']
        tweet_retweet_count = data['retweet_count']
        tweet_number_of_chars = len(tweet_text)
        tweet_number_of_words = len(tweet_text.split())
        user_favourites_count = data['user']['favourites_count']
        user_followers_count = data['user']['followers_count']
        user_creation_date = data['user']['created_at']
        user_creation_date = datetime.strptime(user_creation_date, '%a %b %d %H:%M:%S %z %Y')
        nouns = nouns / tweet_number_of_words
        verbs = verbs / tweet_number_of_words
        adjectives = adjectives / tweet_number_of_words
        adverbs = adverbs / tweet_number_of_words
        interjections = interjections / tweet_number_of_words
        pronouns = pronouns / tweet_number_of_words

        try:
            if train_data[tweet_id] != 'unverified':
                labels_array_test.append(train_data[tweet_id])
                tweet_id_array_test.append(tweet_id)
                tweet_favorite_count_array_test.append(tweet_favorite_count)
                tweet_retweet_count_array_test.append(tweet_retweet_count)
                tweet_number_of_chars_array_test.append(tweet_number_of_chars)
                tweet_number_of_words_array_test.append(tweet_number_of_words)
                user_favourites_count_array_test.append(user_favourites_count)
                user_followers_count_array_test.append(user_followers_count)
                user_date_difference_array_test.append((tweet_creation_date - user_creation_date).days)
                nouns_array_test.append(nouns)
                verbs_array_test.append(verbs)
                adjectives_array_test.append(adjectives)
                adverbs_array_test.append(adverbs)
                interjections_array_test.append(interjections)
                pronouns_array_test.append(pronouns)
                retweeted_array_test.append(retweeted)
                if train_data[tweet_id] != 'true':
                    number_true = number_true + 1
                if train_data[tweet_id] != 'false':
                    number_false = number_false + 1
        except:
            wrongnumber = wrongnumber + 1



#rearange arrays to numpy
numpy_tweet_favorite_count_array_test = np.array(tweet_favorite_count_array_test)
numpy_tweet_retweet_count_array_test = np.array(tweet_retweet_count_array_test)
numpy_tweet_number_of_chars_array_test = np.array(tweet_number_of_chars_array_test)
numpy_tweet_number_of_words_array_test = np.array(tweet_number_of_words_array_test)
numpy_user_favourites_count_array_test = np.array(user_favourites_count_array_test)
numpy_user_followers_count_array_test = np.array(user_followers_count_array_test)
numpy_user_date_difference_array_test = np.array(user_date_difference_array_test)
numpy_nouns_array_test = np.array(nouns_array_test)
numpy_verbs_array_test = np.array(verbs_array_test)
numpy_adjectives_array_test = np.array(adjectives_array_test)
numpy_adverbs_array_test = np.array(adverbs_array_test)
numpy_interjections_array_test = np.array(interjections_array_test)
numpy_pronouns_array_test = np.array(pronouns_array_test)
numpy_retweeted_array_test = np.array(retweeted_array_test)

x = np.column_stack((
                     numpy_pronouns_array_test,
                     numpy_interjections_array_test,
                     numpy_adverbs_array_test,
                     numpy_nouns_array_test,
                     numpy_verbs_array_test,
                     numpy_adjectives_array_test,
                     numpy_user_favourites_count_array_test,
                     numpy_user_followers_count_array_test,
                     numpy_tweet_favorite_count_array_test,
                     numpy_tweet_retweet_count_array_test,
                     numpy_tweet_number_of_chars_array_test,
                     numpy_tweet_number_of_words_array_test,
                     numpy_user_date_difference_array_test,
                     numpy_retweeted_array_test))


y = np.array(labels_array_test)
svm_score = svm_class.score(x, y)
linear_regression_score = linear_regression_class.score(x, y)
tree_score = tree_class.score(x, y)
naive_bayes_score = naive_bayes_class.score(x, y)
nn_score = nn_class.score(x, y)
kn_score = kn_class.score(x, y)
print("svm score: ")
print(svm_score)
print("linear regression score: ")
print(linear_regression_score)
print("tree score: ")
print(tree_score)
print("naive bayes score: ")
print(naive_bayes_score)
print("nn score")
print(nn_score)
print("kn score")
print(kn_score)