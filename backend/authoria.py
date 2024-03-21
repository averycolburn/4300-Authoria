#from Cocktail lab previous repo https://github.com/tl676/Cocktail-Lab/blob/master/backend/cocktailLab.py
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
import numpy as np

class Authoria:
    def __init__(self):
      """Dictionary of {author: genres}"""
      self.authors_to_genre = self.read_file_genre('data/seven_k_books.csv')

      """Dictionary of {authors: descriptions}"""
      self.authors_to_descriptions = self.read_file_description('data/seven_k_books.csv')

      """Dictionary of {authors: average rating}"""
      self.authors_to_ratings = self.read_file_popularity('data/seven_k_books.csv')

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
      print(self.author_index_to_name)
      """List of authors"""
      self.author_names = self.authors_to_genre.keys()
      print(self.author_names)
      """The sklearn TfidfVectorizer object"""
      self.descriptions_tfidf_vectorizer = self.make_vectorizer(binary=True)
      print(self.descriptors_tfidf_vectorizer)
      self.descriptions = [self.authors_to_descriptions[author] for author in
                        self.authors_to_descriptions]
    
      """The term-document matrix"""
      self.description_doc_by_vocab = self.descriptions_tfidf_vectorizer.fit_transform(
            self.descriptions).toarray()

      """Dictionary of {index: token}"""
      self.index_to_vocab = {i: v for i, v in enumerate(
            self.descriptions_tfidf_vectorizer.get_feature_names())}


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
    
    def make_vectorizer(self, binary=False, max_df=1.0, min_df=1, use_stop_words=True):
        """ Returns a TfidfVectorizer object with the above preprocessing properties.

        By default this function returns a tf-idf matrix vectorizer. 
        This can be switched to a binary representation by setting the binary param 
        to True.

        Parameters:
        binary: bool (Default = False)
            A flag to switch between tf-idf representation and binary representation
        max_df: float (Default = 1.0)
            The maximum document frequency to use for the matrix, as a proportion of 
            docs.
        min_df: float or int (Default = 1)
            The miniumum document frequency to use for the matrix. If [0.0,1.0], 
            the parameter represents a proportion of documents, otherwise in absolute
            doc counts. 
        use_stop_words: bool (Default = True)
            A flag to let sklearn remove common stop words.

        Returns:
        A #doc x #vocab np array vectorizer

        """
        if binary:
            use_idf = False
            norm = None
        else:
            use_idf = True
            norm = 'l2'

        if use_stop_words:
            stop_words = 'english'
        else:
            stop_words = None

        tf_mat = TfidfVectorizer(max_df=max_df, min_df=min_df,
                                 stop_words=stop_words, use_idf=use_idf,
                                 binary=binary, norm=norm,
                                 #  analyzer='word', token_pattern='[^,]+'
                                 )

        return tf_mat
    
    def cos_sim(self, vec1, vec2):
        """ Returns the cos sim of two vectors.

        Helper for cos_rank
        """
        num = np.dot(vec1, vec2)
        den = (np.linalg.norm(vec1)) * (np.linalg.norm(vec2))
        if den == 0:
          return 0
        else:
          num / den

    def cos_rank(self, query, doc_by_vocab):
        """ Returns a tuple list that represents the doc indexes and their
            similarity scores to the query.

            Known Problems: This needs to be updated to use document ids (when we 
                            make those).

            Params:
            query: ?? by 1 string np array
                The desired ingredients, as computed by make_query
            doc_by_vocab: ?? by ?? np array
                The doc by vocab matrix as computed by make_matrix

            Returns:
                An (int, int) list where list[0] is the doc index, and list[1] is
                the similarity to the query.
        """
        retval = []

        for d in range(len(doc_by_vocab)):
            doc = doc_by_vocab[d]
            sim = self.cos_sim(query, doc)
            # adjust by popularity
            sim += self.authors_to_ratings[self.author_index_to_name[d]] * 0.02
            retval.append([d, sim])
        sorted_list = list(sorted(retval, reverse=True, key=lambda x: x[1]))
        return sorted_list

    def comma_space_split(self, str):
        return [i for i in ",".join(str.split(" ")).split(",") if i]
    
    def make_query(self, tokens, vectorizer, doc_by_vocab):
        """ Returns a query vector made from tokens that matches the term matrix 
            doc_by_vocab.

        Parameters:
        tokens: str list or str set
            The tokens that make up a query. 
        vectorizer: tfidf vectorizer object
            The doc by vocab matrix as computed by make_matrix
        doc_by_vocab: tfidf or boolean matrix
        """
        vocab_to_index = {v: i for i, v in enumerate(
            vectorizer.get_feature_names())}
        retval = np.zeros_like(doc_by_vocab[0])
        for t in tokens:
            try:
                ind = vocab_to_index[t]
                retval[ind] = 1
            except:
                # token not in matrix
                continue
        return retval

    def query(self, text_description):
        # initialize variables
        matrix = self.description_doc_by_vocab

        rank_list = None
        # the list of indices to return (used by boolean and/not)
        idx_list = None
        # initialize as vector of 0s:
        pref_vec = self.make_query(
            [""], self.descriptions_tfidf_vectorizer, matrix)
        cos_rank = None

        # vectorize inputs, if necessry
        if text_description:
            pref_vec = self.make_query(
                [word.strip().lower()
                 for word in self.comma_space_split(text_description)],
                self.descriptions_tfidf_vectorizer,
                matrix)
        # cosine sim:
        cos_rank = self.cos_rank(pref_vec, matrix)
        rank_list = [{
            'author': self.author_index_to_name[i[0]],
            'description': self.authors_to_description[self.author_index_to_name[i[0]]],
            'genres': self.authors_to_genre[self.author_index_to_name[i[0]]],
            'rating': self.authors_to_ratings[self.author_index_to_name[i[0]]],
        } for i in cos_rank]

        matrix = matrix[idx_list]
        rank_list = [
            i for i in rank_list
            if self.author_name_to_index[i['author']] in idx_list]

        return rank_list

if __name__ == "__main__":
    authoria = Authoria()
    print(authoria.authors_to_ratings)