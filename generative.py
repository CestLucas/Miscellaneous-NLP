import re
import random

############################################################
# Section 1: Markov Models
############################################################

def tokenize(text):
    return re.findall(r"[\w]+|[^\s\w]", text)


def ngrams(n, tokens):
    new_tokens = ["<START>"] * (n - 1) + tokens + ["<END>"]
    return [(tuple(new_tokens[i - n + 1: i]), token) for i, token in enumerate(new_tokens) if i >= n - 1]


class NgramModel(object):

    def __init__(self, n):
        self.n = n
        self.context_token = {}
        self.context_count = {}

    def update(self, sentence):

        for context, token in ngrams(self.n, tokenize(sentence)):
            if context in self.context_token:
                token_count = self.context_token.get(context)
                if token in token_count:
                    token_count[token] += 1
                else:
                    token_count[token] = 1
            else:
                self.context_token[context] = {token: 1}

            if context in self.context_count:
                self.context_count[context] += 1
            else:
                self.context_count[context] = 1

    def prob(self, context, token):
        if context in self.context_token:
            token_count = self.context_token[context]
            if token in token_count:
                return float(token_count[token]) / float(self.context_count[context])
            else:
                return 0.0

        return 0.0

    def random_token(self, context):
        # SUM(j=1, i-1)P(tj | context) <= r < SUM(j=1, i)P(tj | context)
        r = random.random()

        if context in self.context_token:
            token_count = self.context_token[context]
            sorted_count = sorted(token_count.keys())

            context_count = self.context_count[context]

            for i, token in enumerate(sorted_count):
                left = float(sum(token_count[j] for j in sorted_count[:i])) / float(context_count)
                right = left + float(token_count[sorted_count[i]]) / float(context_count)
                if left <= r < right:
                    return token

        return None

    def random_text(self, token_count):
        if self.n == 1:
            return " ".join([self.random_token(()) for __ in range(token_count)])
        else:
            text = []
            context = ("<START>",) * (self.n - 1)

            for i in range(token_count):
                token = self.random_token(context)
                text.append(token)

                if token != "<END>":
                    context = context[1:] + (token,)
                else:
                    context = ("<START>",) * (self.n - 1)  # back to start

            return " ".join(text)

    def perplexity(self, sentence):
        # PRODUCT(i=1, m+1)(1 / p(Wi | Wi-1)) ** (1 / m + 1)
        sentence = tokenize(sentence)
        m = len(sentence)

        product = 1.0
        for context, token in ngrams(self.n, sentence):
            product *= 1 / self.prob(context, token)

        return product ** (float(1) / float(m + 1))


def create_ngram_model(n, path):
    fp = open(path, 'r+')
    model = NgramModel(n)

    for line in fp:
        model.update(line)

    return model
