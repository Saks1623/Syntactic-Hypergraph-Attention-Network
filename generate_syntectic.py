import torch
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm
import spacy
from nltk.stem import WordNetLemmatizer
import os
import pickle


def generate_syntactic_hyperedges(doc_content_list, vocab_dic):
    """
    Returns: List[ Dict[int, List[int]] ]
       Each element is a dict for one document:
         edge_id -> [word_id, word_id, ...]
    """
    print("Generating syntactic hyperedges per document (edge_id -> [word_ids])")
    nlp = spacy.load('en_core_web_sm')
    stemmer = WordNetLemmatizer()
    per_doc_hyperedges = []

    for doc_idx, doc in enumerate(tqdm(doc_content_list, desc="Processing documents")):
        edge_to_words = defaultdict(list)
        edge_counter = 0

        for sentence in doc:
            text = ' '.join(sentence)
            parsed = nlp(text)

            for token in parsed:
                if token.dep_ == 'punct':
                    continue
                children = list(token.children)
                if not children:
                    continue

                # build the group (head + all its direct dependents)
                group = { stemmer.lemmatize(token.text.lower()) }
                for c in children:
                    group.add(stemmer.lemmatize(c.text.lower()))

                # map to vocab IDs, drop out‐of‐vocab words
                group_ids = [vocab_dic[w] for w in group if w in vocab_dic]
                if len(group_ids) < 2:
                    continue

                # record one hyperedge: edge_counter → this list of word_ids
                edge_to_words[edge_counter] = group_ids
                edge_counter += 1


        #print(f"Doc {doc_idx}: formed {edge_counter} syntactic hyperedges")
        per_doc_hyperedges.append(dict(edge_to_words))

    return per_doc_hyperedges

'''
def generate_syntactic_hyperedges(doc_content_list, vocab_dic):
    print("Generating syntactic hyperedges per document (word_id -> [local edge_ids])")
    nlp = spacy.load('en_core_web_sm')
    stemmer = WordNetLemmatizer()

    per_doc_syntactic_hyperedges = []

    for doc_idx, doc in tqdm(enumerate(doc_content_list), desc="Processing documents"):
        syntactic_dict = defaultdict(list)
        edge_counter = 0  # reset for each document

        for sentence in doc:
            text = ' '.join(sentence)
            parsed = nlp(text)

            for token in parsed:
                if token.dep_ == 'punct':
                    continue
                if len(list(token.children)) == 0:
                    continue  # only head words with dependents

                group = set()
                group.add(stemmer.lemmatize(token.text.lower()))

                for child in token.children:
                    group.add(stemmer.lemmatize(child.text.lower()))
                
                # Map to word IDs in vocab
                group_ids = [vocab_dic[word] for word in group if word in vocab_dic]

                if len(group_ids) >= 2:
                    for word_id in group_ids:
                        syntactic_dict[word_id].append(edge_counter)
                    edge_counter += 1
                         

        #print(f"Doc {doc_idx}: {len(syntactic_dict)} syntactic nodes, {edge_counter} unique hyperedges")
        per_doc_syntactic_hyperedges.append(dict(syntactic_dict))

    return per_doc_syntactic_hyperedges
        


def generate_syntactic_hyperedges(doc_content_syntax, contextual_embs, vocab_dic, thres=0.3):
    print("Generating syntactic hyperedges using contextual BERT embeddings...")
    nlp = spacy.load('en_core_web_sm')
    stemmer = WordNetLemmatizer()

    per_doc_syntactic_hyperedges = []

    for doc_idx, (doc, doc_embeds) in tqdm(enumerate(zip(doc_content_syntax, contextual_embs)), desc="Processing documents"):
        syntactic_dict = defaultdict(list)
        edge_counter = 0  # reset for each document

        for sent_idx, sentence in enumerate(doc):
            text = ' '.join(sentence)
            parsed = nlp(text)

            word2pos = {stemmer.lemmatize(tok.text.lower()): i for i, tok in enumerate(parsed)}
            sent_embeds = doc_embeds[sent_idx]

            for token in parsed:
                if token.dep_ == 'punct':
                    continue
                if len(list(token.children)) == 0:
                    continue  # only head words with dependents

                group = set()
                group.add(stemmer.lemmatize(token.text.lower()))
                for child in token.children:
                    group.add(stemmer.lemmatize(child.text.lower()))

                valid_ids = []
                valid_embs = []

                for word in group:
                    if word in vocab_dic and word in word2pos:
                        word_id = vocab_dic[word]
                        pos = word2pos[word]
                        if pos < len(sent_embeds):
                            embedding = sent_embeds[pos]
                            valid_ids.append(word_id)
                            valid_embs.append(embedding)

                if len(valid_ids) >= 2:
                    sims = []
                    for i in range(len(valid_embs)):
                        for j in range(i + 1, len(valid_embs)):
                            cos_sim = F.cosine_similarity(
                                valid_embs[i].unsqueeze(0),
                                valid_embs[j].unsqueeze(0),
                                dim=1
                            ).item()
                            sims.append(cos_sim)

                    avg_sim = sum(sims) / len(sims)
                    if avg_sim >= thres:
                        for word_id in valid_ids:
                            syntactic_dict[word_id].append(edge_counter)
                        edge_counter += 1

        per_doc_syntactic_hyperedges.append(dict(syntactic_dict))

    return per_doc_syntactic_hyperedges

'''
def load_or_generate_hyperedges(dataset,doc_content_list, vocab_dic):
    file_path = os.path.join('data', f'{dataset}_syn_edges.p')

    if os.path.exists(file_path):
        print(f"Loading syntactic hyperedges from {file_path}")
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    else:
        print("Generating syntactic hyperedges...")
        data = generate_syntactic_hyperedges(doc_content_list, vocab_dic)
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved syntactic hyperedges to {file_path}")

    return data