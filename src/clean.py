import pandas as pd


df = pd.read_csv('./data/IMDB Dataset.csv')


print(df.head())


with open("./data/sample_reviews.txt",'w',encoding='utf-8') as f:
    for review in df['review']:
        f.write(review.replace('\n','') + '\n')





