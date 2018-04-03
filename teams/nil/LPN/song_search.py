import fastText as ft
import sys
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

num_search_results = 3

# # load the corresponding song and print it.
print("Loading the dataset...")
songs = pd.read_csv('../dataset/dataset.csv')

print("Loading the embedding model for querying...")
model = ft.load_model("./embeddings_lyrics.bin")
dim = model.get_dimension()

print("Loading the embeddings for songs to compare...")
embedding_of_lyrics = pd.read_csv('../database/song_embedings.csv').values[1:, 1:]

print("Making the KDTree tree...")
tree = KDTree(embedding_of_lyrics)

while(True):
    query = input("Enter your query : ")
    tokens = ft.tokenize(query)
    #print(tokens)

    # embedding matrix for the query
    query_embedding_matrix = []
    for x in tokens:
        query_embedding_matrix.append(model.get_word_vector(x))


    # embedding of the query - using avg method with fastText
    embedding_of_query = model.get_word_vector(query)


    # finding the num_search_results closest songs
    # # finding the index
    dist, ind = tree.query([embedding_of_query], k=num_search_results)
    print(ind)

    # printing the songs
    print("The related songs are...")
    for i in ind:
        artist = songs['artist'][i]
        song = songs['song'][i]
        text = songs['text'][i]
        genre = songs['genre'][i]
        year = songs['year'][i]
        print("The song "+song+" sung by "+artist + "\n" + text)
        print("-----------------------------------")







