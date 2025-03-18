import requests
from bs4 import BeautifulSoup
from transformers import pipeline

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

print("Using Wikipedia as source")

topic = input('What do you want to know about: ')

url = f"https://en.wikipedia.org/wiki/{topic}"

response = requests.get(url, headers=headers)

soup = BeautifulSoup(response.text, "html.parser")

# Load pre-trained summarizer model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Input text to summarize
# text = (''.join(list(soup.select('p')[1:7])))
text = ' '.join([p.text for p in soup.select('p')[1:7]]) 

# Generate summary
summary = summarizer(text, max_length=50, min_length=20, do_sample=False)

# Print the summarized text
print("\nSummarized Text:\n", summary[0]['summary_text'])
