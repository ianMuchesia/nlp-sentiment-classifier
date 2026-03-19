import pandas as pd
from tokenizer import Tokenizer




df = pd.read_csv('./data/IMDB Dataset.csv')


print(df.head())


with open("./data/clean_dataset.txt",'w',encoding='utf-8') as f:
    for index,row in df.iterrows():
        #clean the text ( remove newlines )
        text = row['review'].replace('\n',' ')
    
        label = row['sentiment'] 
        
        f.write(f"{text}\t{label}\n")
        



def save_split(data_slice, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for index, row in data_slice.iterrows():
            # Clean the text (remove newlines)
            text = row['review']
            # Map sentiment: positive -> 1, negative -> 0
            label = 1 if row['sentiment'] == 'positive' else 0
            # Write to file
            f.write(f"{text}\t{label}\n")



def prepare_data(csv_path,max_length=250):
    #1. Load and shuffle
    df = pd.read_csv(csv_path)
    
    df = df.sample(frac=1,random_state=42).reset_index(drop=True)
    
    #initialize your tokenizer
    tokenizer = Tokenizer()
    tokenizer.load("./data/vocab.json")
    
    processed_data = []
    
    
    for _,row in df.iterrows():
        #2. Encode
        tokens = tokenizer.encode([row['review']])[0]
        
        
        #3. Pre-truncate and Pad
        # Keep the last max_length tokens( the 'verdict' )
        tokens = tokens[-max_length:]
        
        #add zeros to the end if it is too short
        padding_size = max_length - len(tokens)
        
        if padding_size > 0:
            tokens = tokens + [0] * padding_size
        
        label =  1 if  row['sentiment'] == 'positive' else 0
        processed_data.append((tokens,label))
        
        
    return processed_data