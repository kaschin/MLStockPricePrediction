import yfinance as yf
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from textaugment import EDA
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class SentimentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]
        EncodedInput = self.tokenizer.encode_plus(text, padding=True, truncation=True, return_tensors='pt')
        InputIDs = EncodedInput['InputIDs'].squeeze()
        attention_mask = EncodedInput['attention_mask'].squeeze()

        return {'InputIDs': InputIDs, 'attention_mask': attention_mask, 'labels': label}

def Classify_Sentiment(text):
    EncodedInput = tokenizer.encode_plus(text, padding=True, truncation=True, return_tensors='pt')
    InputIDs = EncodedInput['InputIDs'].to(device)
    attention_mask = EncodedInput['attention_mask'].to(device)
    with torch.no_grad():
        Out = model(InputIDs, attention_mask=attention_mask)
        Sentiment = torch.argmax(Out.logits).item()

    return Sentiment

def main():

	STOCK_NAME = "GOOG"
	START_DATE = "2010-01-01"
	END_DATE = "2021-12-31"
	stock_data = yf.download(STOCK_NAME, start=START_DATE, end=END_DATE)
	stock_data['text'] = ''  # Placeholder for actual text data
	stock_data['Sentiment'] = 0.0

	csv_file = 'stock_tweets.csv'
	df = pd.read_csv(csv_file)
	filtered_df = df[df['Stock Name'] == STOCK_NAME]

	# Extract the relevant columns
	tweets = filtered_df['Tweet']
	stock_data['text'] = tweets

	for index, row in stock_data.iterrows():
	    text = row['text']
	    sentiment = Classify_Sentiment(text)
	    stock_data.at[index, 'Sentiment'] = sentiment

	train_data = stock_data.iloc[:800]
	test_data = stock_data.iloc[800:]

	FINBERT = "ProsusAI/finbert"
	tokenizer = AutoTokenizer.from_pretrained(FINBERT)
	model = AutoModelForSequenceClassification.from_pretrained(FINBERT)

	train_texts = train_data['text'].tolist()
	train_labels = train_data['Sentiment'].tolist()

	train_dataset = SentimentDataset(train_texts, train_labels)
	train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model.to(device)

	Optimiser = AdamW(model.parameters(), lr=2e-5)
	criterion = nn.CrossEntropyLoss()
	model.train()

	accumulation_steps = 8
	running_loss = 0.0
	for epoch in range(3):
	    for batch_idx, batch in enumerate(train_loader):
	        InputIDs = batch['InputIDs'].to(device)
	        attention_mask = batch['attention_mask'].to(device)
	        labels = batch['labels'].to(device)
	        Optimiser.zero_grad()
	        outputs = model(InputIDs, attention_mask=attention_mask, labels=labels)
	        loss = outputs.loss
	        logits = outputs.logits
	        loss.backward()

	        if (batch_idx + 1) % accumulation_steps == 0:
	            Optimiser.step()
	            Optimiser.zero_grad()
	        else:
	            for param in model.parameters():
	                param.grad = None

	        running_loss += loss.item()
	        logits = logits.detach()
	        loss = loss.detach()

	    epoch_loss = running_loss / len(train_loader)
	    print(f"Epoch {epoch+1} loss: {epoch_loss}")

	Predictions = []

	for index, row in test_data.iterrows():
	    text = row['text']
	    sentiment = Classify_Sentiment(text)
	    input_features = [sentiment]
	    input_features = torch.tensor(input_features).unsqueeze(0).to(device)
	    predicted_price = model(input_features)
	    predicted_price = predicted_price.item()
	    Predictions.append(predicted_price)

	real_prices = test_data['Close'].tolist()
	dates = test_data.index.tolist()

	plt.plot(dates, real_prices, label='Real Prices')
	plt.plot(dates, Predictions, label='Predicted Prices')
	plt.xlabel('Date')
	plt.ylabel('Stock Price')
	plt.title('Real vs Predicted Stock Prices')
	plt.legend()
	plt.xticks(rotation=45)
	plt.show()

if __name__ == "__main__":
	main()
