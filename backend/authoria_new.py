import numpy as np
import csv 
from typing import List, Tuple, Dict
import math
from nltk.tokenize import TreebankWordTokenizer

def read_file_genre(self, filepath):
  """ Returns a dictionary of format {'author' : 'genre1, genre2'}
        Parameters:
        filepath: path to file
        """
  author_genre_dict = {}
  with open(filepath, 'r', encoding='utf-8') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
      authors = row['authors'].split(', ')
      genre = row['categories']
      for author in authors:
        if author not in author_genre_dict:
            #genre dictionary we use set so genres don't get repeated
            author_genre_dict[author] = set() 
        author_genre_dict[author].add(genre)
  return author_genre_dict

def read_file_description(self, filepath):
  """ Returns a dictionary of format {'author' : 'genre1, genre2'}
        Parameters:
        filepath: path to file
        """
  author_description_dict = {}
  with open(filepath, 'r', encoding='utf-8') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
      authors = row['authors'].split(', ')
      description = row['description']
      for author in authors:
            if author not in author_description_dict:
              author_description_dict[author] = []
            author_description_dict[author].append(description)
  return author_description_dict

def read_file_popularity(self, filepath):
  """ Returns a dictionary of format {'author' : average rating}
        Parameters:
        filepath: path to file
        """
  author_popularity_dict = {}
  author_avg_popularity_dict = {}
  with open(filepath, 'r', encoding='utf-8') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
      authors = row['authors'].split(', ')
      avg_rating = row['average_rating']
      if avg_rating != '':
        popularity = float(row['average_rating'])
      else: popularity = 0
      for author in authors:
        if author not in author_popularity_dict:
          author_popularity_dict[author] = []
        author_popularity_dict[author].append(popularity)
        for key, value in author_popularity_dict.items():
          if len(value) == 0:
            average_value = 0
          else:
            average_value = sum(value)/len(value)
          author_avg_popularity_dict[key] = average_value
  return author_avg_popularity_dict

authors_to_genre = read_file_genre('data/seven_k_books.csv')

"""Dictionary of {authors: descriptions}"""
authors_to_descriptions = read_file_description('data/seven_k_books.csv')

"""Dictionary of {authors: average rating}"""
authors_to_ratings = read_file_popularity('data/seven_k_books.csv')

"""Number of authors"""
num_authors = len(authors_to_genre)

"""Dictionary of {author: index}"""
author_name_to_index = {
  name: index for index, name in enumerate(authors_to_genre.keys())}

"""Dictionary of {index: author name}"""
author_index_to_name = { v: k for k, v in author_name_to_index.items()}

"""List of authors"""
author_names = authors_to_genre.keys()

"""List of descriptions in enumerated order """
descriptions = [authors_to_descriptions[author] for author in authors_to_descriptions]

def build_inverted_index(msgs: List[dict]) -> dict:
    """Builds an inverted index from the messages.

    Arguments
    =========

    msgs: list of dicts.
        Each message in this list already has a 'toks'
        field that contains the tokenized message.

    Returns
    =======

    inverted_index: dict
        For each term, the index contains
        a sorted list of tuples (doc_id, count_of_term_in_doc)
        such that tuples with smaller doc_ids appear first:
        inverted_index[term] = [(d1, tf1), (d2, tf2), ...]

    Example
    =======

    >> test_idx = build_inverted_index([
    ...    {'toks': ['to', 'be', 'or', 'not', 'to', 'be']},
    ...    {'toks': ['do', 'be', 'do', 'be', 'do']}])

    >> test_idx['be']
    [(0, 2), (1, 2)]

    >> test_idx['not']
    [(0, 1)]

    """
    inverted_dict = dict()
    msg_index = 0
    for msg in msgs:
      toks = msg["toks"]
      tok_msg_tracker = dict.fromkeys(toks, 0) #dict to count number of times each tok appears in the message
      for token in toks:
        tok_msg_tracker[token] += 1
        if token not in inverted_dict:
          inverted_dict[token] = []
      for token in tok_msg_tracker.keys():
        inverted_dict[token].append((msg_index, tok_msg_tracker[token]))
      msg_index += 1
    return inverted_dict

def compute_idf(inv_idx, n_docs, min_df=10, max_df_ratio=0.95):
    """Compute term IDF values from the inverted index.
    Words that are too frequent or too infrequent get pruned.

    Hint: Make sure to use log base 2.

    inv_idx: an inverted index as above

    n_docs: int,
        The number of documents.

    min_df: int,
        Minimum number of documents a term must occur in.
        Less frequent words get ignored.
        Documents that appear min_df number of times should be included.

    max_df_ratio: float,
        Maximum ratio of documents a term can occur in.
        More frequent words get ignored.

    Returns
    =======

    idf: dict
        For each term, the dict contains the idf value.
    """
    idf_trimmed = dict()
    for term in inv_idx.keys():
      term = term.lower()
      df_t = len(inv_idx[term])
      if df_t >= min_df and (df_t/n_docs) <= max_df_ratio:
        idf_t = math.log2(n_docs/ (1+ df_t) )
        idf_trimmed[term] = idf_t

    return idf_trimmed


def compute_doc_norms(index, idf, n_docs):
    """Precompute the euclidean norm of each document.
    index: the inverted index as above

    idf: dict,
        Precomputed idf values for the terms.

    n_docs: int,
        The total number of documents.
    norms: np.array, size: n_docs
        norms[i] = the norm of document i.
    """
    norms = []
    summ_dict = dict.fromkeys(range(n_docs), 0)
    for word_i in index.keys():
      if word_i in idf.keys():
        for tf_tuple in index[word_i]:
          doc = tf_tuple[0]
          tf = tf_tuple[1]
          summ_dict[doc] += (tf*idf[word_i]) ** 2

    for doc in summ_dict.keys():
      norms.append(math.sqrt(summ_dict[doc]))

    return np.array(norms)


def accumulate_dot_scores(query_word_counts: dict, index: dict, idf: dict) -> dict:
    """Perform a term-at-a-time iteration to efficiently compute the numerator term of cosine similarity across multiple documents.

    Arguments
    =========

    query_word_counts: dict,
        A dictionary containing all words that appear in the query;
        Each word is mapped to a count of how many times it appears in the query.
        In other words, query_word_counts[w] = the term frequency of w in the query.
        You may safely assume all words in the dict have been already lowercased.

    index: the inverted index as above,

    idf: dict,
        Precomputed idf values for the terms.
    doc_scores: dict
        Dictionary mapping from doc ID to the final accumulated score for that doc
    """
    doc_scores = dict()
    q_norm = 0
    for query_word in query_word_counts.keys():
      if query_word in idf.keys():
        q_tf = query_word_counts[query_word]
        q_j = q_tf* idf[query_word]
        for doc_tuple in index[query_word]:
          doc = doc_tuple[0]
          d_tf = doc_tuple[1]
          d_ij = d_tf * idf[query_word]
          if doc not in doc_scores.keys():
            doc_scores[doc] = 0
          doc_scores[doc] += d_ij* q_j

    return doc_scores


def index_search(
    query: str,
    index: dict,
    idf,
    doc_norms,
    score_func=accumulate_dot_scores,
    tokenizer=TreebankWordTokenizer(),
) -> List[Tuple[int, int]]:
    """Search the collection of documents for the given query

    Arguments
    =========

    query: string,
        The query we are looking for.

    index: an inverted index as above

    idf: idf values precomputed as above

    doc_norms: document norms as computed above

    score_func: function,
        A function that computes the numerator term of cosine similarity (the dot product) for all documents.
        Takes as input a dictionary of query word counts, the inverted index, and precomputed idf values.
        (See Q7)

    tokenizer: a TreebankWordTokenizer

    Returns
    =======

    results, list of tuples (score, doc_id)
        Sorted list of results such that the first element has
        the highest score, and `doc_id` points to the document
        with the highest score.
    """
    results = []
    query_toks = tokenizer.tokenize(query.lower())
    query_word_count = dict()
    q_norm = 0
    for tok in query_toks:
      if tok not in query_word_count.keys():
        query_word_count[tok] = 0
      query_word_count[tok] += 1
    for i in query_word_count.keys():
      if i in idf.keys():
        tf_i = query_word_count[i]
        idf_i = idf[i]
        q_norm += (tf_i * idf_i) ** 2
    q_norm = math.sqrt(q_norm)

    dot_scores = score_func(query_word_count, index, idf)
    for doc_id in dot_scores.keys():
      doc_score = dot_scores[doc_id]/(q_norm*doc_norms[doc_id])
      results.append((doc_score, doc_id))

    results.sort(key=lambda x: x[0], reverse=True)
    return results

def query(query_string):
  flat_msgs = descriptions
  inv_idx = build_inverted_index(flat_msgs)
  idf_dict = compute_idf(inv_idx, len(flat_msgs), min_dt = 5)
  inv_idx = {key: val for key, val in inv_idx.items() if key in idf_dict} # prune the terms left out by idf
  doc_norms = compute_doc_norms(inv_idx, idf_dict, len(flat_msgs))
  query = query_string
  results = index_search(query, inv_idx, idf_dict, doc_norms)
  return results
    