import math

############################################################
# Section 1: Hidden Markov Models
############################################################


def load_corpus(path):
    fp = open(path, "r+")
    return [[tuple(token.split("=")) for token in line.split()] for line in fp]


class Tagger(object):

    def __init__(self, sentences):
        self.tag_count = {"DET": 0, "NOUN": 0, "ADJ": 0, "VERB": 0, "ADP": 0, "ADV": 0, "PRON": 0, ".": 0, "NUM": 0,
                          "CONJ": 0, "PRT": 0, "X": 0}

        init_tag_count = {"DET": 0, "NOUN": 0, "ADJ": 0, "VERB": 0, "ADP": 0, "ADV": 0, "PRON": 0, ".": 0, "NUM": 0,
                          "CONJ": 0, "PRT": 0, "X": 0}

        token_of_tag_count = {"DET": {}, "NOUN": {}, "ADJ": {}, "VERB": {}, "ADP": {}, "ADV": {}, "PRON": {}, ".": {},
                              "NUM": {}, "CONJ": {}, "PRT": {}, "X": {}}

        previous_tag_of_tag_count = {"DET": {}, "NOUN": {}, "ADJ": {}, "VERB": {}, "ADP": {}, "ADV": {}, "PRON": {},
                                     ".": {}, "NUM": {}, "CONJ": {}, "PRT": {}, "X": {}}

        self.transition_prob = {"DET": {}, "NOUN": {}, "ADJ": {}, "VERB": {}, "ADP": {}, "ADV": {}, "PRON": {},
                                ".": {}, "NUM": {}, "CONJ": {}, "PRT": {}, "X": {}}

        self.emission_prob = {"DET": {}, "NOUN": {}, "ADJ": {}, "VERB": {}, "ADP": {}, "ADV": {}, "PRON": {},
                              ".": {}, "NUM": {}, "CONJ": {}, "PRT": {}, "X": {}}

        prev_tag = None
        for sentence in sentences:
            init_tag_count[sentence[0][1]] += 1

            for token, tag in sentence:
                self.tag_count[tag] += 1

                token_count = token_of_tag_count[tag]
                if token in token_count:
                    token_count[token] += 1
                else:
                    token_count[token] = 1

                if prev_tag is not None:
                    tag_count = previous_tag_of_tag_count[prev_tag]
                    if tag in tag_count:
                        tag_count[tag] += 1
                    else:
                        tag_count[tag] = 1

                prev_tag = tag

        smoothing = 1e-5
        num_of_sentences = len(sentences)

        # initial tag prob: π(ti) for 1 ≤ i ≤ n
        self.initial_tag_prob = {tag: float(init_tag_count[tag] + smoothing) /
                                 float(num_of_sentences + smoothing * (num_of_sentences + 1))
                                 for tag in init_tag_count.keys()}

        # transition prob: a(ti → tj) for 1 ≤ i, j ≤ n
        # emission prob: b(ti → wj) for 1 ≤ i ≤ n and 1 ≤ j ≤ m
        for tag in token_of_tag_count.keys():
            tag_count = self.tag_count[tag]

            # transition
            transition_prob_tag = self.transition_prob[tag]
            transition_prob_tag["<UNK>"] = float(smoothing) / float(tag_count + smoothing * (num_of_sentences + 1))

            p_tag_of_tag = previous_tag_of_tag_count[tag]
            for cur_tag in p_tag_of_tag.keys():
                transition_prob_tag[cur_tag] = float(p_tag_of_tag[cur_tag]) / \
                                               float(tag_count + smoothing * (num_of_sentences + 1))

            # emission
            token_of_tag = token_of_tag_count[tag]
            num_tokens_of_tag = len(token_of_tag)

            emission_prob_tag = self.emission_prob[tag]
            emission_prob_tag["<UNK>"] = float(smoothing) / float(tag_count + smoothing * (num_tokens_of_tag + 1))

            for cur_tag in token_of_tag_count[tag].keys():
                emission_prob_tag[cur_tag] = float(token_of_tag[cur_tag]) / \
                                       float(tag_count + smoothing * (num_tokens_of_tag + 1))

    def most_probable_tags(self, tokens):
        # i* = argmaxi b(ti → wj)
        tags = []

        for token in tokens:
            max_prob = 0
            max_tag = None

            for tag in self.emission_prob.keys():
                tag_prob = self.emission_prob[tag]
                if token in tag_prob:
                    if tag_prob[token] > max_prob:
                        max_prob = tag_prob[token]
                        max_tag = tag

            tags.append(max_tag)

        return tags

    def viterbi_tags(self, tokens):
        v_tags = [{}]
        v_path = {}

        first_token = tokens[0]
        for tag in self.tag_count.keys():
            initial_prob = math.log(self.initial_tag_prob[tag])

            if first_token in self.emission_prob[tag]:
                emission_prob = math.log(self.emission_prob[tag][first_token])
            else:
                emission_prob = math.log(self.emission_prob[tag]["<UNK>"])

            v_tags[0][tag] = initial_prob + emission_prob
            v_path[tag] = [tag]

        # reconstruct the path
        num_of_tokens = len(tokens)
        for i in range(1, num_of_tokens):
            v_tags.append({})
            new_path = {}

            for tag in self.tag_count.keys():
                back_path = []
                for prev_tag in self.tag_count.keys():
                    prev_tag_transition_prob = self.transition_prob[prev_tag]
                    tag_emission_prob = self.emission_prob[tag]

                    if tag in prev_tag_transition_prob:
                        transition_prob = math.log(prev_tag_transition_prob[tag])
                    else:
                        transition_prob = math.log(prev_tag_transition_prob["<UNK>"])

                    if tokens[i] in tag_emission_prob:
                        emission_prob = math.log(tag_emission_prob[tokens[i]])
                    else:
                        emission_prob = math.log(tag_emission_prob["<UNK>"])

                    prev_prob = v_tags[i-1][prev_tag] + transition_prob + emission_prob
                    back_path.append((prev_prob, prev_tag))

                best_prob, best_tag = max(back_path)
                v_tags[i][tag] = best_prob
                new_path[tag] = v_path[best_tag] + [tag]

            v_path = new_path

        best_prob, best_tag = max((v_tags[num_of_tokens - 1][tag], tag) for tag in self.tag_count.keys())
        return v_path[best_tag]
