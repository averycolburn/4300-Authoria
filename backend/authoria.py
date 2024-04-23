import csv 
from typing import List, Tuple, Dict
import math
from nltk.tokenize import TreebankWordTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
# from scipy.sparse.linalg import svds
from collections import defaultdict 

class Authoria:
  def __init__(self):
      """Dictionary of {author: genres}"""
      self.authors_to_genre = self.read_file_genre('data/seven_k_books.csv')

      """Dictionary of {book title (string): description (string)}"""
      self.book_to_descrip = self.read_book_descrip('data/seven_k_books.csv')

      """Dictionary of {authors: descriptions}"""
      self.authors_to_descriptions = self.read_file_description('data/seven_k_books.csv')

      """Dictionary of {authors: book titles}"""
      self.authors_to_books = self.read_file_books('data/seven_k_books.csv')      
      
      """Dictionary of {authors: average rating}, {authors: weighted rating}"""
      self.authors_to_ratings , self.authors_to_weighted_ratings = self.read_file_popularity('data/seven_k_books.csv')

      """Number of authors"""
      self.num_authors = len(self.authors_to_genre)

      """Dictionary of {author: index}"""
      self.author_name_to_index = {
            name: index for index, name in
            enumerate(self.authors_to_genre.keys())
        }

      """Dictionary of {index: author name}"""
      self.author_index_to_name = {
            v: k for k, v in self.author_name_to_index.items()}

      """List of authors"""
      self.author_names = self.authors_to_genre.keys()

      """List of descriptions"""
      self.descriptions = [self.authors_to_descriptions[author] for author in
                        self.authors_to_descriptions]
      
      """Set of common words"""
      self.common = self.read_common_words('data/unigram_freq.csv',100) #pick number of common words to exclude here

      """Set of closest words using SVD"""
      self.vectorizer, self.td_matrix = self.vectorize_descriptions('data/seven_k_books.csv')
      


  def read_common_words(self, filepath, n):
    """ Returns a set of the n most common words in the dataset at filepath
          Parameters:
          filepath: path to file
          n: number of common words to exclude
    """
    common = set()
    with open(filepath, 'r', encoding='utf-8') as file:
      csv_reader = csv.DictReader(file)
      for x in range(n):
        row = next(csv_reader)
        # print(row["word"])
        common.add(row['word'])
        x+=1
    return common
      
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
  
  def read_book_descrip(self, filepath):
    """ Returns a dictionary of format {'author' : 'genre1, genre2'}
          Parameters:
          filepath: path to file
          """
    book_descrip_dict = {}
    with open(filepath, 'r', encoding='utf-8') as file:
      csv_reader = csv.DictReader(file)
      for row in csv_reader:
        title = row['title']
        descrip = row['description']
        book_descrip_dict[title] = descrip
    return book_descrip_dict

  def read_file_books(self, filepath):
    """ Returns a dictionary of format {'author' : 'descriptions'}
          Parameters:
          filepath: path to file
        Descriptions is a string list.
    """
    author_title_dict = {}
    with open(filepath, 'r', encoding='utf-8') as file:
      csv_reader = csv.DictReader(file)
      for row in csv_reader:
        authors = row['authors'].split(', ')
        title = row['title']
        for author in authors:
              if author not in author_title_dict:
                author_title_dict[author] = []
              author_title_dict[author].append(title)
    return author_title_dict
  
  def read_file_description(self, filepath):
    """ Returns a dictionary of format {'author' : book titles}
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
    """ Returns a 2 dictionaries of format {'author' : average rating}, {'author' : average weighted rating}
          Parameters:
          filepath: path to file
          """
    author_popularity_dict = {}
    avg_popularity_dict = {} #sofia 3/21
    author_avg_popularity_dict = {}
    author_weighted_avg_dict = {} #sofia 3/21
    with open(filepath, 'r', encoding='utf-8') as file:
      csv_reader = csv.DictReader(file)
      for row in csv_reader:
        authors = row['authors'].split(', ')
        avg_rating = row['average_rating']
        r_c=0 #ratings count is automatically 0 unless weighted #sofia 3/21
        ratings_count = row['ratings_count']
        if ratings_count!="": #sofia 3/21
          r_c = float(ratings_count) #sofia 3/21
        if avg_rating != '':
          popularity = float(row['average_rating']) 
          w_pop = popularity + r_c/100 #number of review for this book #sofia 3/21
        else: popularity , w_pop = 0,0 #edited by sofia 3/21, can remove w_pop to remove sofia edit
        for author in authors: 
          if author not in author_popularity_dict: 
            author_popularity_dict[author] = []
            avg_popularity_dict[author] = []
          author_popularity_dict[author].append(popularity)
          avg_popularity_dict[author].append(w_pop)
          for key, value in author_popularity_dict.items():
            if len(value) == 0:
              average_value = 0
            else:
              average_value = sum(value)/len(value)
            author_avg_popularity_dict[key] = average_value    
          #sofia 3/21
          for key, value in avg_popularity_dict.items():
            if len(value) == 0:
              average_value = 0
            else:
              average_value = sum(value)/len(value)
            author_weighted_avg_dict[key] = average_value
    return author_avg_popularity_dict, author_weighted_avg_dict

  def build_inverted_index(self, msgs: List[dict]) -> dict:
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
        toks = [x.lower() for x in toks]
        tok_msg_tracker = dict.fromkeys(toks, 0) #dict to count number of times each tok appears in the message
        for token in toks:
          tok_msg_tracker[token] += 1
          if token not in inverted_dict:
            inverted_dict[token] = []
        for token in tok_msg_tracker.keys():
          inverted_dict[token].append((msg_index, tok_msg_tracker[token]))
        msg_index += 1
      return inverted_dict

  def compute_idf(self, inv_idx, n_docs, min_df=5, max_df_ratio=0.9):
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


  def compute_doc_norms(self, index, idf, n_docs):
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

      return (norms)


  def accumulate_dot_scores(self, query_word_counts: dict, index: dict, idf: dict) -> dict:
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
      k = 1/1000 #k is weight for impact of author popularity
      doc_scores = dict()
      word_scores=defaultdict(dict)
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
            if(query_word in word_scores[doc]):
               word_scores[doc][query_word]+=d_ij*q_j
            else:
               word_scores[doc][query_word]=d_ij*q_j
            # word_scores[doc][query_word]+= d_ij*q_j
            doc_scores[doc] += d_ij* q_j+ self.authors_to_weighted_ratings[self.author_index_to_name[doc]]*k #sofia 3/21

      return doc_scores, word_scores

 
  def index_search(
      self,
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
      # dot_scores, words=score_func(self, query_word_count, index, idf)
      # highest_contributors=defaultdict(int)
      query_toks = tokenizer.tokenize(query.lower())
      query_word_count = dict()
      q_norm = 0
      for tok in query_toks:
        if not (tok in self.common): #sofia 4/14
          if tok not in query_word_count.keys():
            query_word_count[tok] = 0
          query_word_count[tok] += 1
      for i in query_word_count.keys():
        if i in idf.keys():
          tf_i = query_word_count[i]
          idf_i = idf[i]
          q_norm += (tf_i * idf_i) ** 2
          # highest_contributors[i]+= (tf_i * idf_i) ** 2
      q_norm = math.sqrt(q_norm)
      dot_scores,  words = score_func(self, query_word_count, index, idf)
      for doc_id in dot_scores.keys():
        doc_score = dot_scores[doc_id]/(q_norm*doc_norms[doc_id])
        common_words=words[doc_id]
        sort_words=sorted(common_words, key=common_words.get, reverse=True)
        results.append((doc_score, doc_id, sort_words[0:2]))
      # highest_contributors.sort(key=lambda x: x[1], reverse=True)
      results.sort(key=lambda x: x[0], reverse=True)
      return results

  def query(self,query_string : str):
    
    flat_msgs = []
    for description in self.descriptions:
      descript_toks = TreebankWordTokenizer().tokenize(description[0])
      flat_msgs.append({"toks" : descript_toks})
    inv_idx = self.build_inverted_index(flat_msgs)
    idf = self.compute_idf(inv_idx, len(flat_msgs), min_df = 5)
    inv_idx = {key: val for key, val in inv_idx.items() if key in idf} # prune the terms left out by idf
    doc_norms = self.compute_doc_norms(inv_idx, idf, len(flat_msgs))
    ranked_results = self.index_search(query_string, inv_idx, idf, doc_norms)
    rank_list = [] 
    for i in ranked_results: 
      author_name = self.author_index_to_name[i[1]]
      book_lst = self.authors_to_books[author_name]
      author_profile = {
            'author': author_name,
            'titles' : self.authors_to_books[author_name],
            'genres': self.authors_to_genre[author_name],
            'rating': self.authors_to_ratings[author_name],
            'score':round(i[0]*100,2),
            'common': i[2],
            'feature_title': book_lst[0],
            'feature_descrip': self.book_to_descrip[book_lst[0]]
        }
      rank_list.append(author_profile)
    return rank_list
  
  def vectorize_descriptions(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            descriptions = [row['description'] for row in csv_reader]

        vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, min_df=75)
        td_matrix = vectorizer.fit_transform(descriptions)
        return vectorizer, td_matrix

def closest_words(self, word_in, k=10):
    word_to_index = self.vectorizer.vocabulary_
    index_to_word = {i: t for t, i in word_to_index.items()}

    if word_in not in word_to_index:
      return "Not in vocab."

    word_index = word_to_index[word_in]
    sims = self.td_matrix.dot(self.td_matrix[word_index, :].T).toarray().ravel()
    asort = sorted(range(len(sims)),key=lambda i: (-sims)[i])[:k + 1]
    return [(index_to_word[i], sims[i]) for i in asort[1:]]



  
if __name__ == "__main__":
    authoria = Authoria()