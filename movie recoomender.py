
import pandas as pd  
from string import punctuation,digits
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import  NearestNeighbors
from difflib import get_close_matches


org_dataset = pd.read_csv(r"E:\dataset\imdb_top_1000.csv")

org_dataset = org_dataset.drop("Poster_Link",axis =1)

dataset = org_dataset.loc[:,["Genre","Director","Star1","Star2","Star3","Star4"]]

def remove_extra(data):
    string = ""
    for i in data:
        if i not in punctuation and i not in digits:
            string +=i
    return string

for col_name in dataset.columns:
    dataset[col_name]=dataset[col_name].apply(remove_extra)

new_data = " "
for col_name in dataset.columns:
    new_data = new_data + dataset[col_name] + " "
dataset=pd.DataFrame(new_data,columns=["data"])

vector = TfidfVectorizer()
vector.fit(dataset['data'])
dataset=vector.transform(dataset['data']).toarray()


model_nn = NearestNeighbors(n_neighbors=5,metric= 'cosine')
model_nn.fit(dataset)

movie_name = input("Enter movie name: ").lower()
movie_list = org_dataset["Series_Title"].str.lower().tolist()

match = get_close_matches(movie_name, movie_list, n=1, cutoff=0.5)

if not match:
    print("Movie not found!")
else:
    matched_movie = match[0]
    print("Did you mean:", matched_movie)
    index = movie_list.index(matched_movie)

distances, indices = model_nn.kneighbors([dataset[index]])
rec_indices = indices[0][1:]  
print("\nRecommended Movies:\n")
for i in rec_indices:
    print(org_dataset.iloc[i]["Series_Title"])








