import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.title("Spotify Songs EDA")
st.markdown('Spotify is a music streaming service that allows users to listen to music on their mobile devices, computers, and other smart speakers. The service is available in 79 countries and has over 286 million users, including 130 million paying subscribers. The company was founded in 2006 and is headquartered in Stockholm, Sweden. The dataset that I have chosen contains 114,000 rows, and 19 columns. Each represents one song from spotify and contains information about the song, artist, album, and a bunch of other metrics (e.g. danceability, energy, acousticness). The dataset was collected by a user on Kaggle and can be found here: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset. The first five rows of the dataset are provided below.')

url = 'https://raw.githubusercontent.com/ayang903/spotify/main/dataset.csv'
df = pd.read_csv(url, index_col=[0])
#df = pd.read_csv('dataset.csv', index_col=[0])
df.head()
df.info()
df.describe()
st.dataframe(df.head())
st.markdown("""---""")

# Data Cleaning
df = df.dropna() #drop 2 null values
df = df.drop_duplicates(
  subset = ['artists', 'track_name'],
  keep = 'first').reset_index(drop = True)
#some artists release the same songs twice, on different albums, drop the re-release

# Correlation Matrix
fig = px.imshow(df.corr())
fig.update_layout(title_text = 'Pairwise Correlation of Features')
st.plotly_chart(fig)


# 1. Top Artists by average popularity
genres = [
    'anime',
    'pop',
    'k-pop',
    'hip-hop',
    'edm',
    'rock',
    'country',
    'chill',
    'r-n-b',
    'disney',
    'dance',
    'jazz',
    'study',
    'electronic',
    'classical',
    'spanish'
]

fig = make_subplots(rows=4, cols=4, vertical_spacing=0.16, subplot_titles = genres)
row=1
col=1

for genre in genres:
    top_artists = df.query('track_genre == @genre')
    top_artists = top_artists.groupby(by='artists').mean().sort_values('popularity', ascending=False)
    top_artists = top_artists.loc[~top_artists.index.str.contains(';')]
    top_artists = top_artists.head(10)
    
    fig.add_trace(
    go.Bar(name=genre, x=top_artists.index, y=top_artists['popularity']),
    row=row, col=col)
    
    col += 1
    if (col == 5):
        col = 1
        row += 1
    
    fig.update_layout(height=2000, width=1000, title_text="10 Most Popular Artists in Specific Genres, by Average Popularity")
    
st.plotly_chart(fig)


# 2. Most popular genres by average popularity
top_genres = df.groupby('track_genre').mean().sort_values('popularity', ascending=False)
fig = px.bar(top_genres, x=top_genres.index, y='popularity')
fig.update_layout(title_text='Most Popular Genres by Average Popularity')
st.plotly_chart(fig)

# ## 3. Most diverse artists by number of genres and number of songs
artists = df.groupby('artists').nunique().sort_values('track_genre', ascending=False)
average_num_genres_per_artists = np.mean(artists['track_genre'])
artists = artists.reset_index(level=0)
artists = artists.head(10)
print(f'The average number of genres per artist is {average_num_genres_per_artists}')

artists = artists[['artists', 'track_genre', 'track_id']]
artists =  artists.rename({'track_genre':'genre_count', 'track_id':'track_count'}, axis=1)

fig = px.bar(artists, x='artists', y='genre_count', color='track_count',  color_continuous_scale=px.colors.sequential.Viridis)
fig.update_layout(title_text='Most Diverse Artists and Their Number of Songs')
st.plotly_chart(fig)


#4. boxplot for danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence
cols = [
    'popularity',
    'duration_ms',
    'danceability',
    'energy'
]

fig = make_subplots(rows=2, cols=2, vertical_spacing=0.16, subplot_titles = cols)
row=1
colum=1

for col in cols:
    
    fig.add_trace(
    go.Box(name=col, y=df[col]), row=row, col=colum)
    
    colum += 1
    if (colum == 3):
        colum = 1
        row += 1
    
    fig.update_layout(height=2000, width=1000, title_text="Boxplots for Four Numerical Features (any more and it lags my computer)")
     
st.plotly_chart(fig)
    
    
# # 5. Predict how the __danceability__ depends on other properties using linear regression as baseline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


y = df['danceability']
X = df[['energy', 'liveness', 'valence', 'tempo']]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)
lr = LinearRegression()
lr.fit(X_train, y_train)
pred = lr.predict(X_test)


coeff_df = pd.DataFrame(lr.coef_, X.columns, columns=['Coefficient'])
#coeff_df
# seems like valence is our biggest indicator of danceability, correlation heatmap backs us up


from sklearn import metrics
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, pred)))


df1 = pd.DataFrame({'Actual': y_test, 'Predicted': pred})
df1 = df1.head(25)


fig = go.Figure(data=[
    go.Bar(name='Actual Danceability', x=df.index, y=df1['Actual']),
    go.Bar(name='Predicted Danceablity', x=df.index, y=df1['Predicted'])
])
# Change the bar mode
fig.update_layout(title_text = 'Actual vs Predicted Danceability', barmode='group', bargap=0.5)
st.plotly_chart(fig)

st.header('Conclusion')
st.markdown('Our correlation heatmap tells us that the energy and loudness of a song are very positively correlated with each other and the acousticness and energy of a song are very negatively correlated. Some other noteworthy correlations are: ')
st.markdown('- Loudness vs Acousticness')
st.markdown('- Loudness vs Instrumentalness')
st.markdown('- Danceability vs Valence')
st.markdown('- Speechiness vs Explicit')
st.markdown('\n')
st.markdown('We can also easily see the top 10 artists of each genre from the matrix of barplots')
st.markdown('The genre that has the highest average popularity is k-pop while romance music is the least popular genre. Looks like "IVE" is the most popular artist from the most popular genre, on average.')
st.markdown('The most diverse artist is Sumo, with 5 different genres and 16 songs. The average number of genres per artist is 1.05, meaning most artists typically stay in only one genre')
st.markdown('Our boxplots tell us a lot about the spread of our data and shows us the outliers. For example, popularity scores have a median at 35, meaning half the songs have a popularity greater than 35 and half less. There also seem to be a few **really** long songs, the maximum being ~80 minutes (after looking these up, they all seem to be "mixes", which are just collections of multiple songs, all put into one "mix")!')
st.markdown('Finally, our regression model tells us that the *valence* of a song is the biggest indicator of danceability, followed by energy, liveness, and tempo. This is just our baseline model with an RMSE of 0.152 and r2 score of 0.268, meaning there is a lot of room for improvement. If I were to extend this project, my next steps would be to try to improve our model by adding more features, using a different model, regularize, and try an ensemble of models. Another option would be to try to classify the genre of a song based on the features we have using a decision tree or KNN model.')

