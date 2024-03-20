#from Cocktail lab previous repo https://github.com/tl676/Cocktail-Lab/blob/master/backend/cocktailLab.py
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import csv
import math


class Authoria:
    def __init__(self):
      """Dictionary of {author: genres}"""
      self.authors_to_genre = self.read_file_genre('data/seven_k_books.csv')

      """Dictionary of {authors: descriptions}"""
      self.authors_to_genre = self.read_file_description('data/seven_k_books.csv')

      """Number of authors"""
      self.num_authors = len(self.cocktail_names_to_ingreds)

      """Dictionary of {author: index}"""
      self.author_name_to_index = {
            name: index for index, name in
            enumerate(self.author_to_genre.keys())
        }

      """Dictionary of {index: cocktail name}"""
      self.author_index_to_name = {
            v: k for k, v in self.author_name_to_index.items()}

      """List of cocktail names"""
      self.author_names = self.author_names_to_ingreds.keys()

      """The sklearn TfidfVectorizer object"""
      self.descriptions_tfidf_vectorizer = self.make_vectorizer(binary=True)

      self.descriptions = [self.author_names_to_descriptions[author] for author in
                        self.author_names_to_descriptions]

      """The term-document matrix"""
      self.description_doc_by_vocab = self.descriptions_tfidf_vectorizer.fit_transform(
            self.descriptions).toarray()

      """Dictionary of {index: token}"""
      self.index_to_vocab = {i: v for i, v in enumerate(
            self.descriptions_tfidf_vectorizer.get_feature_names())}

      self.rocchio_alpha = 1.0

      self.rocchio_beta = 1.0


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
                genre = row['genre']
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
                popularity = row['average_rating']
                for author in authors:
                    if author not in author_popularity_dict:
                        author_popularity_dict[author] = []
                    author_popularity_dict[author].append(popularity)
            for key, value in author_popularity_dict.items():
              if len(value) == 0:
                average_value = 0
              else:
                avergage_value = sum(value)/len(value)
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