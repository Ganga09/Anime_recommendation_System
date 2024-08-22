Anime Recommendation System: 

> recommends 10 related animes to the input title based on title, type and genre.

This project is an Anime recommendation system designed to suggest 10 anime titles closely related to a user's input. The system analyzes the input title and utilizes Cosine Similarity to measure the textual similarity between the input and other anime titles, types, and genres in the database.

Calculating the cosine of the angle between the vector representations of the titles, a curated list of 10 recommendations. This approach ensures that the suggested anime shares thematic and content-related similarities with the input title, providing users with personalized and accurate recommendations.

This project demonstrates the application of natural language processing and similarity algorithms in enhancing user experience in content discovery.


Dataset: https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database

Anime.csv

1. anime_id - myanimelist.net's unique id identifying an anime.
2. name - Full name of anime.
3. genre - comma-separated list of genres for this anime.
4. type - movie, TV, OVA,Special,ONA etc.
5. episodes - how many episodes in this show and 1 for movie.
6. rating - an average rating out of 10 for this anime.
7. members - number of community members that are in this anime's
"group".
