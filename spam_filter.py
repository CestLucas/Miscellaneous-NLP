import email
import math
import collections
import os


############################################################
# Section 1: Spam Filter
############################################################

def load_tokens(email_path):
    fp = open(email_path)
    msg = email.message_from_file(fp)
    fp.close()

    tokens = []
    for line in email.iterators.body_line_iterator(msg):
        tokens.extend(line.split())
    return tokens


def log_probs(email_paths, smoothing):
    # merge all email tokens into a single word list
    words = []
    for path in email_paths:
        words.extend(load_tokens(path))

    # count up vocab/words
    counts = collections.Counter()
    for word in words:
        counts[word] += 1

    total_words = len(words)
    total_vocab = len(counts)

    # finally add unknown token
    counts['<UNK>'] = 0

    # calculate probabilities
    probs = dict()
    for token in counts:
        probs[token] = math.log((counts[token] + smoothing) / (total_words + smoothing * (total_vocab + 1)))

    return probs


class SpamFilter(object):

    def __init__(self, spam_dir, ham_dir, smoothing):
        # list all spam/ham paths
        spam_paths = [os.path.join(spam_dir, s) for s in os.listdir(spam_dir)]
        ham_paths = [os.path.join(ham_dir, h) for h in os.listdir(ham_dir)]

        self.stats_spam = log_probs(spam_paths, smoothing)
        self.stats_ham = log_probs(ham_paths, smoothing)

        spam_size = len(spam_paths)
        ham_size = len(ham_paths)

        prob_spam = float(spam_size) / float(spam_size + ham_size)
        prob_not_spam = 1 - prob_spam

        # take the log
        self.prob_spam = math.log(prob_spam)
        self.prob_not_spam = math.log(prob_not_spam)

    def is_spam(self, email_path):
        prob_spam = self.prob_spam
        prob_not_spam = self.prob_not_spam

        tokens = load_tokens(email_path)

        for token in tokens:
            if token in self.stats_spam:
                prob_spam += self.stats_spam[token]
            else:
                prob_spam += self.stats_spam['<UNK>']

            if token in self.stats_ham:
                prob_not_spam += self.stats_ham[token]
            else:
                prob_not_spam += self.stats_ham['<UNK>']

        return True if prob_spam > prob_not_spam else False

    def most_indicative_spam(self, n):
        common_words = list(set(self.stats_spam).intersection(self.stats_ham))

        lst = []
        for word in common_words:
            val = self.stats_spam[word] - math.log(math.exp(self.stats_spam[word]) * math.exp(self.prob_spam)
                                                   + math.exp(self.stats_ham[word]) * math.exp(self.prob_not_spam))
            lst.append((word, val))

        lst.sort(key=lambda x: x[1], reverse=True)
        return [i[0] for i in lst[:n]]

    def most_indicative_ham(self, n):
        common_words = list(set(self.stats_spam).intersection(self.stats_ham))

        lst = []
        for word in common_words:
            val = self.stats_ham[word] - math.log(math.exp(self.stats_spam[word]) * math.exp(self.prob_spam)
                                                  + math.exp(self.stats_ham[word]) * math.exp(self.prob_not_spam))
            lst.append((word, val))

        lst.sort(key=lambda x: x[1], reverse=True)
        return [i[0] for i in lst[:n]]
