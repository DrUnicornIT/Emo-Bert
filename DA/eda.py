import random
import re
import pandas
import csv
from nltk.corpus import wordnet
from random import shuffle

random.seed(1)

# stop_words_list
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our',
              'ours', 'ourselves', 'you', 'your', 'yours',
              'yourself', 'yourselves', 'he', 'him', 'his',
              'himself', 'she', 'her', 'hers', 'herself',
              'it', 'its', 'itself', 'they', 'them', 'their',
              'theirs', 'themselves', 'what', 'which', 'who',
              'whom', 'this', 'that', 'these', 'those', 'am',
              'is', 'are', 'was', 'were', 'be', 'been', 'being',
              'have', 'has', 'had', 'having', 'do', 'does', 'did',
              'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
              'because', 'as', 'until', 'while', 'of', 'at',
              'by', 'for', 'with', 'about', 'against', 'between',
              'into', 'through', 'during', 'before', 'after',
              'above', 'below', 'to', 'from', 'up', 'down', 'in',
              'out', 'on', 'off', 'over', 'under', 'again',
              'further', 'then', 'once', 'here', 'there', 'when',
              'where', 'why', 'how', 'all', 'any', 'both', 'each',
              'few', 'more', 'most', 'other', 'some', 'such', 'no',
              'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
              'very', 's', 't', 'can', 'will', 'just', 'don',
              'should', 'now', '']

char_valid = 'qwertyuiopasdfghjklzxcvbnm '


# cleaning up text
def get_only_chars(line):
    clean_line = ""
    line = line.replace("’", "")
    line = line.replace("'", "")
    line = line.replace("-", " ")  # replace hyphens with spaces
    line = line.replace("\t", " ")
    line = line.replace("\n", " ")
    line = line.lower()

    for char in line:
        if char in char_valid:
            clean_line += char
        else:
            clean_line += ' '

    clean_line = re.sub(' +', ' ', clean_line)
    if clean_line[0] == ' ':
        clean_line = clean_line[1:]
    return clean_line


# This function to get synonyms of word
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in char_valid])
            synonyms.add(synonym)
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)


def synonym_replacement(words, sdp_words, n):
    new_words = words.copy()

    random_words = list(set([word for word in words if word not in stop_words + sdp_words]))

    random.shuffle(random_words)
    num_replaced = 0
    for random_word in random_words:
        synonyms = get_synonyms(random_word)
        if len(synonyms) > 0:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            print("replaced", random_word, "with", synonym)
            num_replaced += 1
        if num_replaced == n:
            break
    sentence = ' '.join(new_words)
    return sentence


# Randomly delete words from the sentence with probability p
def random_deletion(words, sdp_words, p):
    if len(words) == 1:
        return words

    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p or word in sdp_words:
            new_words.append(word)

    if len(new_words) == 0:
        rand_int = random.randint(0, len(words) - 1)
        return [words[rand_int]]
    sentence = ' '.join(new_words)
    return sentence


def swap_word(new_words, sdp_word_idx):
    random_idx_first = random.randint(0, len(new_words) - 2)
    random_idx_second = random_idx_first
    num_swapped = 0
    while random_idx_first == random_idx_second or random_idx_first in sdp_word_idx or random_idx_second in sdp_word_idx:
        random_idx_first = random.randint(0, len(new_words) - 2)
        random_idx_second = random.randint(0, len(new_words) - 2)
        num_swapped += 1
        if num_swapped > 30:
            return new_words

    new_words[random_idx_first], new_words[random_idx_second] = new_words[random_idx_second], new_words[
        random_idx_first]
    return new_words


# Randomly swap two words in the sentence n times
def random_swap(words, sdp_word_idx, n):
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words, sdp_word_idx)
    sentence = ' '.join(new_words)
    return sentence


def add_word(new_words):
    synonyms = []
    counter = 0
    while len(synonyms) < 1:
        random_word = new_words[random.randint(0, len(new_words) - 1)]
        synonyms = get_synonyms(random_word)
        counter += 1
        if counter >= 10:
            return
    random_synonym = synonyms[0]
    random_idx = random.randint(0, len(new_words) - 1)
    new_words.insert(random_idx, random_synonym)


# Randomly insert n words into the sentence
def random_insertion(words, n):
    new_words = words.copy()
    for _ in range(n):
        add_word(new_words)
    return new_words


# Randomly delete words from the sentence with probability p
def random_deletion(words, sdp_words, p):
    if len(words) == 1:
        return words

    # randomly delete words with probability p
    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p or word in sdp_words:
            new_words.append(word)

    # if you end up deleting all words, just return a random word
    if len(new_words) == 0:
        rand_int = random.randint(0, len(words) - 1)
        return [words[rand_int]]
    sentence = ' '.join(new_words)
    return sentence


def eda(sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, alpha_rd=0.1, num_aug=9):
    sentence = get_only_chars(sentence)
    words = sentence.split(' ')
    words = [word for word in words if word != '']
    num_words = len(words)

    augmented_sentences = []
    num_new_per_technique = int(num_aug / 4) + 1

    # synonym replacement
    if (alpha_sr > 0):
        num_sr = max(1, int(alpha_sr * num_words))
        for _ in range(num_new_per_technique):
            a_words = synonym_replacement(words, [], num_sr)
            augmented_sentences.append(' '.join(a_words))

    # random insert
    if (alpha_ri > 0):
        num_ri = max(1, int(alpha_ri * num_words))
        for _ in range(num_new_per_technique):
            a_words = random_insertion(words, num_ri)
            augmented_sentences.append(' '.join(a_words))

    # random swap
    if (alpha_rs > 0):
        num_rs = max(1, int(alpha_rs * num_words))
        for _ in range(num_new_per_technique):
            a_words = random_swap(words, [], num_rs)
            augmented_sentences.append(' '.join(a_words))

    # random delete
    if (alpha_rd > 0):
        for _ in range(num_new_per_technique):
            a_words = random_deletion(words, [], alpha_rd)
            augmented_sentences.append(' '.join(a_words))

    # augmented_sentences = [get_only_chars(sentence) for sentence in augmented_sentences]
    shuffle(augmented_sentences)

    # if num_aug > 0:
    #    augmented_sentences = augmented_sentences[:num_aug]
    # else:
    # keep_prob = num_aug / len(augmented_sentences)
    # augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

    augmented_sentences.append(sentence)
    return augmented_sentences


data = pandas.read_csv('../REdata/DAIR/CL/data_cl.csv')
count_sentence = [0, 0, 0, 0, 0, 0]
header = ['label', 'RS', 'RD', 'SR', 'O']

batch_size = 1000
with open('../REdata/DAIR/CL/train.csv',mode='w',newline='',buffering=1024*1024*1024) as file:
    writer = csv.writer(file)
    writer.writerow(header)
    

    batch = []
    for i in range(len(data['text'])):

        sentence = data['text'][i]
        label = data['label'][i]
        words = sentence.split(' ')
        if (len(words) > 2):
            dt = [label,''.join(random_swap(words, [], 1)),''.join(random_deletion(words, [], 0.1)),''.join(synonym_replacement(words, [], 1)),sentence]
            batch.append(dt)
        if len(batch) == batch_size or i == len(data['text']) - 1:
            if len(batch) > 0:
                writer.writerows(batch)
                batch = []