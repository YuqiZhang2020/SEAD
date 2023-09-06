import collections
import os
import time
import itertools
import numpy as np
import config
import nltk
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


def read_languages(paths):
    languages = dict()
    for corpus_filename in os.listdir(os.path.join(paths.data_root, 'corpus')):
        corpus_path = os.path.join(paths.data_root, 'corpus', corpus_filename)
        with open(corpus_path) as f:
            language = f.readlines()
            language = [s.strip().split() for s in language]
            language = [s[1:-1] for s in language]
            languages[os.path.splitext(corpus_filename)[0]] = language
    return languages


def read_durations(paths):
    durations = dict()
    for duration_filename in os.listdir(os.path.join(paths.data_root, 'duration')):
        duration_path = os.path.join(paths.data_root, 'duration', duration_filename)
        with open(duration_path) as f:
            duration = f.readlines()
            durations.update({(os.path.splitext(duration_filename)[0].split("_")[0], r_d.split()[0]): float(r_d.split()[1]) for r_d in duration})   
    return durations


def get_pcfg(rules):
    root_rules = list()
    non_terminal_rules = list()
    grammar_rules = list()
    for rule in rules:
        tokens = rule.split()
        for i in range(len(tokens)):
            token = tokens[i]
            if token[0] == 'E':
                tokens[i] = tokens[i].replace('E', 'OR')
            elif token[0] == 'P':
                tokens[i] = tokens[i].replace('P', 'AND')
        rule = ' '.join(tokens)

        if rule.startswith('S'):
            root_rules.append(rule)
        else:
            non_terminal_rules.append(rule)

    for k, v in collections.Counter(root_rules).items():
        grammar_rules.append(k + ' [{}]'.format(float(v) / len(root_rules)))

    grammar_rules.extend(non_terminal_rules)
    return grammar_rules


def read_induced_grammar(paths):
    # Read grammar into nltk
    grammar_dict = dict()
    for or_grammar_file in os.listdir(os.path.join(paths.data_root, 'grammar')):
        with open(os.path.join(paths.data_root, 'grammar', or_grammar_file)) as f:
            rules = [rule.strip() for rule in f.readlines()]
            grammar_rules = get_pcfg(rules)
            grammar = nltk.PCFG.fromstring(grammar_rules)
            grammar_dict[os.path.splitext(or_grammar_file)[0]] = grammar
    return grammar_dict


def get_production_prob(selected_edge, grammar):
    # Find the corresponding production rule of the edge, and return its probability
    for production in grammar.productions(lhs=selected_edge.lhs()):
        if production.rhs() == selected_edge.rhs():
            return production.prob()


def find_parent(selected_edge, chart):
    # Find the parent edges that lead to the selected edge
    p_edges = list()
    for p_edge in chart.edges():
        # Important: Note that p_edge.end() is not equal to p_edge.start() + p_edge.dot(),
        # when a node in the edge spans several tokens in the sentence
        if p_edge.end() == selected_edge.start() and p_edge.nextsym() == selected_edge.lhs():
            p_edges.append(p_edge)
    return p_edges


def get_edge_prob(selected_edge, chart, grammar):
    # Compute the probability of the edge by recursion
    prob = get_production_prob(selected_edge, grammar)
    if selected_edge.start() != 0:
        parent_prob = 0
        for parent_edge in find_parent(selected_edge, chart):
            parent_prob += get_edge_prob(parent_edge, chart, grammar)
        prob *= parent_prob
    return prob


def remove_duplicate(tokens):
    return [t[0] for t in itertools.groupby(tokens)]


def predict_next_symbols(grammar, tokens):
    tokens = remove_duplicate(tokens)
    symbols = list()
    
    earley_parser = nltk.EarleyChartParser(grammar, trace=0)
    try:
        e_chart = earley_parser.chart_parse(tokens)
    
    except ValueError:
        return list()
    end_edges = list()

    for edge in e_chart.edges():
        if edge.end() == len(tokens):
            # Only add terminal nodes
            if isinstance(edge.nextsym(), str):
                symbols.append(edge.nextsym())
                end_edges.append(edge)

    probs = list()
    for end_edge in end_edges:
        probs.append(get_edge_prob(end_edge, e_chart, grammar))
 
    # Eliminate duplicate
    symbols_no_duplicate = list()
    probs_no_duplicate = list()
    for s, p in zip(symbols, probs):
        if s not in symbols_no_duplicate:
            symbols_no_duplicate.append(s)
            probs_no_duplicate.append(p)
        else:
            probs_no_duplicate[symbols_no_duplicate.index(s)] += p

    return symbols_no_duplicate, probs_no_duplicate


def lcs(valid_tokens, tokens):
    lengths = [[0 for j in range(len(tokens) + 1)] for i in range(len(valid_tokens) + 1)]
    # row 0 and column 0 are initialized to 0 already
    for i, x in enumerate(valid_tokens):
        for j, y in enumerate(tokens):
            if x == y:
                lengths[i+1][j+1] = lengths[i][j] + 1
            else:
                lengths[i+1][j+1] = max(lengths[i+1][j], lengths[i][j+1])

    x, y = len(valid_tokens), len(tokens)
    matched_tokens = None
    while x != 0 and y != 0:
        if lengths[x][y] == lengths[x-1][y]:
            x -= 1
        elif lengths[x][y] == lengths[x][y-1]:
            y -= 1
        else:
            assert valid_tokens[x-1] == tokens[y-1]
            matched_tokens = valid_tokens[:x]
            for i in range(x, len(tokens)):
                if i < len(valid_tokens):
                    matched_tokens.append(valid_tokens[i])
                else:
                    matched_tokens.append(valid_tokens[-1])
            break

    if not matched_tokens:
        if len(valid_tokens) < len(tokens):
            matched_tokens = valid_tokens[:]
            for _ in range(len(valid_tokens), len(tokens)):
                matched_tokens.append(valid_tokens[-1])
        else:
            matched_tokens = valid_tokens[:len(tokens)]
    return len(tokens) - lengths[-1][-1], matched_tokens


def find_closest_tokens(language, tokens, truncate=False):
    min_distance = np.inf
    best_matched_tokens = None

    for valid_tokens in language:
        d, matched_tokens = lcs(valid_tokens, tokens)
        if d < min_distance:
            min_distance = d
            if not truncate:
                best_matched_tokens = matched_tokens
            else:
                best_matched_tokens = matched_tokens[:len(valid_tokens)]

    return min_distance, best_matched_tokens


def encode_sg(sgs, object_classes, relationship_classes, MAX_SEQUENCE_LENGTH):
    v_x = []
    for sg in sgs.strip().split(";"):
        f_x = np.zeros([len(relationship_classes), len(object_classes)], np.float16)
        orps = sg.split(",")
        for orp in orps:
            if orp.replace("\n", "").strip() != "":
                o_ = orp.split(":")[0]
                sr_ = orp.split(":")[1].split("/")[0]
                cr_ = orp.split(":")[1].split("/")[1]
                f_x[relationship_classes.index(sr_), object_classes.index(o_)] = 1
                f_x[relationship_classes.index(cr_), object_classes.index(o_)] = 1
        v_x.append(f_x)
    v_x = np.asarray(v_x)
    v_x = v_x[np.newaxis, :, :, :]
    v_x = pad_sequences(v_x, maxlen=MAX_SEQUENCE_LENGTH)
    return v_x


def scene_predict(paths, obs_sg, t):
    grammar_dict = read_induced_grammar(paths)
    duration_dict = read_durations(paths)
    languages = read_languages(paths)
    S = {}
    v = obs_sg.strip().split(";")
    for i in range(len(v)):
        for j in range(len(v[i])):
            orps = v[i][j].split(",")
            for orp in orps:
                if orp != "":
                    o_ = orp.split(":")[0]
                    sc_ = orp.split(":")[1]
                    if o_ not in S:
                        S[o_] = [sc_]
                    elif sc_ != S[o_][-1]:
                        S[o_].append(sc_)
    pre_sg = ""
    for k in S.keys():
        # obtain the stochastic grammar for object k
        grammar = grammar_dict[k]
        language = languages[k]
        # obtain the duration of spatial-contact events for object k
        key = k + "_duration"
        duration = duration_dict[key]
        s = " ".join(S[k])
        if s.split()[-1] not in duration:
            d, matched_tokens = find_closest_tokens(language, s.split()[-1])
            tn = duration[matched_tokens[0]]
        else:
            tn = duration[s.split()[-1]]
        while(tn<t):
            tokens = s.split()
            d, matched_tokens = find_closest_tokens(language, tokens)
            lr, pr = predict_next_symbols(grammar, matched_tokens)
            res = lr[pr.index(max(pr))]
            pre_osr = k + ":" + res
            tn += duration[res]
            s += (" " + res)
            pre_sg += pre_osr + ","
    return pre_sg


def main():
    paths = config.Paths()
    start_time = time.time()

    MAX_SEQUENCE_LENGTH = 4
    t = 3

    relationship_classes = []
    with open('./annotations/relationship_classes.txt', 'r') as f:
        for line in f.readlines():
            relationship_classes.append(line.strip())
    relationship_classes[0] = 'looking_at'
    relationship_classes[1] = 'not_looking_at'
    relationship_classes[5] = 'in_front_of'
    relationship_classes[7] = 'on_the_side_of'
    relationship_classes[10] = 'covered_by'
    relationship_classes[11] = 'drinking_from'
    relationship_classes[13] = 'have_it_on_the_back'
    relationship_classes[15] = 'leaning_on'
    relationship_classes[16] = 'lying_on'
    relationship_classes[17] = 'not_contacting'
    relationship_classes[18] = 'other_relationship'
    relationship_classes[19] = 'sitting_on'
    relationship_classes[20] = 'standing_on'
    relationship_classes[25] = 'writing_on'

    object_classes = ['__background__']
    with open('./annotations/object_classes.txt', 'r') as f:
        for line in f.readlines():
            object_classes.append(line.strip())
    object_classes[9] = 'closet/cabinet'
    object_classes[11] = 'cup/glass/bottle'
    object_classes[23] = 'paper/notebook'
    object_classes[24] = 'phone/camera'
    object_classes[31] = 'sofa/couch'

    model = load_model("action_anticipation_model.h5")

    results = []
    with open('relation_test.txt') as f:
        lines = f.readlines()
        for line in lines:
            pre_sg = scene_predict(paths, line, t)
            all_sg = line + pre_sg
            
            tx = encode_sg(all_sg, object_classes, relationship_classes, MAX_SEQUENCE_LENGTH)
            tx = tx.reshape(tx.shape[1], tx.shape[2]*tx.shape[3])
            tx = tx.astype('float16')
            res = model.predict(np.array([tx]))
            results.append(res[0]) # Can be utilized for calculating mean Average Precision
    
    print('Time elapsed: {}'.format(time.time() - start_time))

if __name__ == '__main__':
    main()