#Anime Recommendation system:


Anime Recommendation System :
recommends 10 related animes to the input title based on title, type and genre.

## Load data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fuzzywuzzy import process
import warnings
warnings.filterwarnings('ignore')
anime=pd.read_csv('/content/drive/MyDrive/Project_dataset/anime.csv')
anime

## DATA:

*   7 columns: 'anime_id', 'name', 'genre', 'type', 'episodes', 'rating', 'members'

*  rows=12294



print("Dataset, a glimpse:","\n Head\n",anime.head(),"\Tail: \n",anime.tail())

print("Shape: ",anime.shape)
print("Columns:", anime.columns)

Check for null

anime.isnull().sum()

Check datatypes

anime.dtypes

anime.info()

sns.heatmap(anime.corr(numeric_only=True))

#count of unique values in genre, type and episodes
lst=['genre','type','episodes']
for i in lst:
  print("\n.....................................................",i,"....................................................\n")
  print("VALUE:",anime[i].unique())
  print("COUNT:",anime[i].value_counts())



anime['episodes']=anime['episodes'].replace('Unknown',np.nan)
anime['episodes']=anime['episodes'].astype(float)

sns.countplot(x='type',data=anime,color='g')
plot=plt.gca()
plot.set_title('Type of anime',fontsize=20,color='brown')

TV_anime=anime[anime['type']=='TV']
TV_anime['genre'].value_counts().sort_values(ascending=True).tail(20).plot.barh(figsize=(8,8),color='r')
plt.title('TV-Anime genre',fontsize=15,color='g')
plt.xlabel('frequency')
plt.ylabel('genres')
plt.show()

#analysing the top 10 animes with respect to number of members
top=anime.sort_values(by='members',ascending=False).drop_duplicates(subset='name').head(10)
palette=sns.color_palette('deep',len(top))
plt.bar(top['name'],top['members'],color=palette)

plt.title("Top 10 Anime by Number of members")
plt.xlabel('Anime Title')
plt.ylabel('Number of members')
plt.xticks(rotation=40,ha='right')
plt.show()

plt.figure(figsize=(15,5))
sns.boxplot(x='type',y='rating',data=anime)
plt.title('Anime-type V/S Rating',fontsize=15, color='m')
plt.show()

#fill missing values for genre type and rating
anime['rating']=anime['rating'].fillna(anime['rating'].mean())
anime['genre']=anime['genre'].fillna(anime['genre'].mode()[0])
anime['type']=anime['type'].fillna(anime['type'].mode()[0])

anime.isnull().sum()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

x=anime.iloc[0]
x

## Get column by concatenating genre and type

def getString(row):
    result_string = row.genre+", "+row.type  # genre and type attributes of the row with a comma and space in between.
    result_string=result_string.replace(', ',' ')
    return  result_string
anime['string'] = anime.apply(getString,axis=1) #applies the get_string function to each row of the anime DataFrame to assigns the result to a new column named string in the anime DataFrame
anime.head()

## Vectorization: TF-IDF

tfidf=TfidfVectorizer(max_features=3000)

#fit the tfidf model to the string column and transforms it into a sparse matrix of TF-IDF features. 3000: Dimensionality Reduction,Performance Improvement:

vector=tfidf.fit_transform(anime['string']) #all the anime's the TF-IDF vectors in the vector matrix.
vector.shape

## Anime to index

anime2idx=pd.Series(anime.index,index=anime['name'])#creates a Pandas Series where the index is anime['name'] and the values are there indices
anime2idx

## Define the recommendation function

  


> Recommendation System:
1. The recommended_anime_cosine function looks up the index of the given anime title.
2. If the title is not found, fuzzy matching is used to suggest a similar title.
3. If the index is a Series, the first element is used.
4. The cosine similarity between the TF-IDF vector of the given anime and all other animes is calculated.
5. The top 10 most similar animes are returned based on the similarity scores.

1.Try Block: Attempts to find the index of the provided title.


> **Input**: The function takes the user-provided title and tries to find the closest match to it from the list of anime names (anime['name'].tolist()).


List of Anime Names: ['Naruto', 'One Piece', 'Attack on Titan', ...] etc

2. Except Block: If a KeyError is raised i.e., the title is not found, it uses process.extract to find the closest match.



> **Fuzzy Matching**: It uses fuzzy string matching to compare the title against each anime name in the list.


matches = process.extract("Narutoo", ['Naruto', 'One Piece', 'Attack on Titan', ...], limit=1)

3.Check Match Score: If the similarity score of the closest match is 80 or above, it suggests the similar title. If no good match is found, it returns "Anime not found."
> **Output**: It returns a list of tuples, where each tuple contains a matched name and its corresponding similarity score. Since limit=1, it returns the best match.

[('Naruto', 90)] : 'Naruto' is the closest match with a similarity score of 90.

def recommended_anime_cosine(title):
    try:
        idx=anime2idx[title]
    except:
        matches = process.extract(title, anime['name'].tolist(), limit=1) #fuzzy string matching to handle  inputs that does not exactly match any anime titles in the dataset.
        if matches and matches[0][1] >= 80:
            similar_name = matches[0][0]
            return f"Did you mean '{similar_name}'?"
        print("Anime Not Exist")
        return
    if isinstance(idx, pd.Series):
        idx=idx.iloc[0]
    #This line checks if idx is an instance of pd.Series, which is the data type for a Series in the Pandas library.
    #If idx is a Pandas Series, this line assigns idx the value of the first element in the Series.# iloc[0] is used to access the first element based on its integer position.
    selected_t=vector[idx]
    scores=cosine_similarity(selected_t,vector) #cosine similarity scores
    scores=scores.flatten() #1D array
    recommended_idx=(-scores).argsort()[1:11] # Sorts animes by their similarity to the given anime in descending order,including itself
    return anime['name'].iloc[recommended_idx]

Example:
Negated Scores: [-1.0, -0.95, -0.9, -0.85, ...] :(By default, the argsort() method sorts in ascending order. If you want the highest similarity scores first, need to reverse the order)
Sorted Indices: [0, 1, 2, 3, ...]
Selected Indices: [1, 2, 3, ..., 10] (skip 0)

recommended_anime_cosine('Naruto')

Locally done in pycharm using Streamlit



```
#
from fuzzywuzzy import process
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the anime dataset
anime = pd.read_csv('datasepath of anime.csv')
anime["rating"] = anime["rating"].fillna(anime["rating"].mean())
anime["genre"] = anime["genre"].fillna(anime["genre"].mode()[0])
anime["type"] = anime["type"].fillna(anime["type"].mode()[0])

# Create the string feature for TF-IDF
def get_string(row):
    result_string = row.genre + ", " + row.type
    result_string = result_string.replace(', ', ' ')
    return result_string

anime['string'] = anime.apply(get_string, axis=1)

# Vectorize the string feature
tfidf = TfidfVectorizer(max_features=3000)
vector = tfidf.fit_transform(anime['string'])
#Series
anime2idx=pd.Series(anime.index,index=anime['name'])

#cosinesimilarity
def recommended_anime_cosine(title):
    try:
        idx = anime2idx[title]
    except KeyError:
        matches = process.extract(title, anime['name'].tolist(), limit=1)
        if matches and matches[0][1] >= 80:
            similar_name = matches[0][0]
            return f"Did you mean '{similar_name}'?"
        return "Anime not found."
    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]
    qy = vector[idx]
    scores = cosine_similarity(qy, vector).flatten()
    recommended_idx = (-scores).argsort()[1:11]  
    return anime['name'].iloc[recommended_idx].tolist()

 # Define the Streamlit app
def main():
    st.title('Anime Recommendation System')
    options = anime['name'].tolist()
    text_input = st.selectbox('Select an Anime Title:', options)
    #text_input = st.text_input('Enter title:')
    if st.button('View Recommendations'):
        result = recommended_anime_cosine(text_input)
        with st.expander("Recommendations"):
            if isinstance(result, str):  # Check if the result is a string (error message)
                st.write(result)
            else:
                for anime_title in result:
                    st.write(anime_title)
    st.write("TOP 10 Animes:")
    st.image('path of top animes.png')
    st.write("Types of Animes:")
    st.image('path of types of animes.png')

if __name__ == "__main__":
    main()
```


