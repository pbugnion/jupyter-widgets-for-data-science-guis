
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier


class Classifier(object):

    def __init__(self):
        self.categories = []
        self.category2id = {}
        self.feature_pipeline = None

    def train(self, transactions, categories):
        self.categories = categories
        memos = np.array(
            [transaction.memo for transaction in transactions]
        )
        self.category2id = {
            category: id_ for id_, category
            in enumerate(categories)
        }
        target = [
            self.category2id[transaction.category]
            for transaction in transactions
        ]
        self.feature_pipeline = Pipeline([
            ('vect', CountVectorizer(stop_words='english')),
            ('tfidf', TfidfTransformer()),
            ('clf', SGDClassifier(loss='log'))
        ])
        self.feature_pipeline.fit(memos, target)

    def predict_proba(self, transactions):
        memos = np.array(
            [transaction.memo for transaction in transactions])
        if len(memos):
            probas = self.feature_pipeline.predict_proba(memos)
        else:
            probas = []
        id2category = {v: k for k, v in self.category2id.items()}
        output_probas = []
        for row in probas:
            most_likely_category_id = np.argsort(row)[-1]
            most_likely_category = id2category[most_likely_category_id]
            proba = row[most_likely_category_id]
            output_probas.append((most_likely_category, proba))
        return output_probas

    def predict_one(self, transaction):
        return self.predict_proba([transaction])[0]
