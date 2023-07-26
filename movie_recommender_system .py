#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')


# In[3]:


movies.head(1)


# In[4]:


credits.head(1)


# In[5]:


#credits.head(1)['cast'].values


# In[6]:


#merge two data freame based on title
movies.merge(credits,on='title')


# In[7]:


# movies.merge(credits, on='title').shape


# In[8]:


#movies.shape


# In[9]:


#credits.shape


# In[10]:


movies = movies.merge(credits, on = 'title')


# In[11]:


movies.head(1)


# In[12]:


#making a list of which attributes are useful for recommendation or which is not
#budget, homepage,production_company, production_countries, release_date(because of numeric), revenue, runtime, spoken_language, status, tagline(because we already taken overview), vote_average, vote_count, movie_id(id already taken),  is not useful column
#genres, id(for showing posters), keywords, title, overview, cast, crew is useful
#popularity is not useful from our point of view,since we are creating tags so and it is numeric measure so it can not be fit with our approach


# In[13]:


#movies['original_language'].value_counts()


# In[14]:


#movies.info()


# In[15]:


movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]
movies.head()


# In[16]:


#Data preprocessing


# In[17]:


#To check missing attributes
movies.isnull().sum()


# In[18]:


#Here we can see we have 3 movies whose overview is unknown and since it is not a big number so we can drop these movies.
movies.dropna(inplace=True)


# In[19]:


#To check duplicate data
movies.duplicated().sum()


# In[20]:


movies.iloc[0].genres


# In[21]:


#Above is list of dictionary so now i want to chage the format i.e
#'[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]' 
#['Action','Adventure','Fantasy','SciFi']


# In[22]:


import ast
#ast.literal_eval(obj)


# In[23]:


#We can create a helper function name "convert"
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L    


# In[24]:


#Since it is string so we can not call convert function so now we have to convert this string of list into list for this we have to import module such as
# import ast
# ast.literal_eval


# In[25]:


# import ast
# ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')


# In[26]:


#a = [{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]


# In[27]:


movies['genres'] = movies['genres'].apply(convert)


# In[28]:


movies['keywords'] = movies['keywords'].apply(convert)


# In[29]:


movies.head()


# In[30]:


#movies['cast'][0]


# In[31]:


def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L


# In[32]:


movies['cast'] = movies['cast'].apply(convert3)


# In[33]:


movies.head()


# In[34]:


def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L   


# In[35]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[36]:


movies.head()


# In[37]:


movies['overview'][0] #this is string i want to convert this into list because all column here are in list format so.


# In[38]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[39]:


movies.head() #This is our list format columns.


# In[40]:


# now our task is to make a tag column so for this we just have to connect all the list and change again into strings and we got a paragraph and that's what we want that is the tag column.
#But there is one problem that the space between name i.e look at the cast column there is a name like "sam worthington" i want to remove this space between sam and worthington so that we can make this as "samworthington" for this we have to apply Transformation.


# In[41]:


movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x]) #here we use list comprehension i.e. [i.replace(" ","") for i in x]


# In[42]:


movies.head()


# In[43]:


movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])


# In[44]:


movies.head()


# In[45]:


#Now we make a new column namely Tag and in this column we just connect all these four column genres, keywords, cast and crew


# In[46]:


movies['tags'] = movies['overview'] + movies['genres']+ movies['keywords'] + movies['cast'] + movies['crew']


# In[47]:


movies.head()


# In[48]:


#Here now we have a tags column of all four column so we don't need overview,genres,keywords.cast and crew column so we just remove all these and make new dataframe.
new_df = movies[['movie_id','title','tags']]


# In[49]:


new_df #here in the tags column this is list we just change it into strings so for this we just write a code as shown below


# In[50]:


new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))


# In[51]:


new_df.head()


# In[52]:


get_ipython().system(' pip install nltk')


# In[53]:


import nltk


# In[54]:


from nltk.stem.porter import PorterStemmer #PorterStemmer is class
ps = PorterStemmer()


# In[55]:


#below shown is a helper function namely "stem"
def stem(text): #we got here a text
    y = [] #cresting list
    
    for i in text.split():  #then we split that is we change the string into list
        y.append(ps.stem(i))  #now we stem everyword.......and here we store this into y so we used here 'y.append'
    return " ".join(y)  #now here we again change this list into string.


# In[56]:


new_df['tags'] = new_df['tags'].apply(stem)


# In[57]:


new_df['tags'][0] #Here this is strings


# In[58]:


#we want to change all these into lower case because of suggestion of recommenders...
new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())


# In[59]:


new_df.head()


# In[60]:


#Now we need to do vectorization because we want to check similar movies on user's choice and since this is a texual data so this is the tough task so to make it easy we need to convert this text into vector from vectorization.
#This process is known as "Textvectorization". using the techniques "Bag of words"


# In[61]:


#bag of words tech.: - It is basically connnect all words(like- tag1+tag2+...) in given paragraph(i.e.in Tags column),and then perform vectorization tech on those words
#When we perform vectorization then we just remove stop words(i.e. that words which have meaning in sentence forming but in the actual sentence meaning does not effect just like are, of, in , a, etc...) )
#other than that words we perform vectorization on them.


# In[62]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000, stop_words='english')


# In[63]:


vectors = cv.fit_transform(new_df['tags']).toarray()


# In[64]:


vectors[0]


# In[65]:


cv.get_feature_names() #when we add all tags then it can be said as corpus


# In[66]:


#Now we have to apply stemming i.e. ['loved', 'loving', 'love'] ---> ['love', 'love', 'love']
#for this we have to apply stemming for this we need a library namel "nltk" as shown above.
#why we need stemming because we don't want same name like action and actions because it doesn't have any contexual meaning in the movie recommendation.(like action and actions are similer mean in a movie.)


# In[67]:


ps.stem('loved')


# In[68]:


stem('In the 22nd century, a paraplegic Marine is dispatched to the moon Pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. Action Adventure Fantasy ScienceFiction cultureclash future spacewar spacecolony society spacetravel futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d SamWorthington ZoeSaldana SigourneyWeaver JamesCameron')


# In[69]:


#So now basically we are in high dimensional space so that's why we use cosine distance(i.e basically based on angle between two vectors...and angle is inversely propotional to distance.) between two vectors instead of euclidean distance measure.


# In[70]:


from sklearn.metrics.pairwise import cosine_similarity   #Here we are using cosine similarity (it is betwenn 0 and 1, 1 means too good similarity 0 means too less similarities)


# In[71]:


similarity = cosine_similarity(vectors)


# In[72]:


similarity[0]   #this is the array 1 that means similarity of 1st movie with all 4806 movies


# In[73]:


sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6] #Now we are sorting the list using enumerate function so that we can sort the list without losing index position of movie. and by using lambda function we check the similarity from 2nd movie which is the nearily similar to that 1st movie because similarity of 1st movie with itself is always 1.


# In[78]:


#Now we build a recommendation function so that it will give me similar 5 movies of given movie by user.
def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]   #we first check index of the given particular movie for matching other movies index
    distances = similarity[movie_index]  #here we calculate distance between angle of vectors and then we sort the array to find that 5 similar movies
    movies_list =  sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(new_df.iloc[i[0]].title)


# In[80]:


recommend('Batman Begins')

