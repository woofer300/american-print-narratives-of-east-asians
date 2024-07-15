import spacy
from textblob import TextBlob
import nltk
from nltk.tokenize import sent_tokenize

# Download necessary NLTK data
nltk.download('punkt')

# Load the spaCy model
nlp = spacy.load('en_core_web_sm')

# The provided text
text = """
San Jose was once home to one of the largest Chinatowns in California. In the heart of downtown, it was the center of life for Chinese immigrants who worked on nearby farms and orchards.

More than a century after arsonists burned it to the ground in 1887, the San Jose City Council on Tuesday unanimously approved a resolution to apologize to Chinese immigrants and their descendants for the role the city played in “systemic and institutional racism, xenophobia, and discrimination.”

San Jose, with a population over 1 million, is the largest city in the country to formally apologize to the Chinese community for its treatment of their ancestors. In May, the city of Antioch, California, apologized for its mistreatment of Chinese immigrants, who built tunnels to get home from work because they were banned from walking the streets after sundown.

“It’s important for members of the Chinese American community to know that they are seen and that the difficult conversations around race and historic inequities include the oppression that their ancestors suffered,” San Jose Mayor Sam Liccardo said.

The apologies come amid a wave of attacks against the Asian community since the pandemic began last year. Other cities, specifically in the Pacific Northwest, have issued apologies in decades past. California, too, apologized in 2009 to Chinese workers and Congress has apologized for the Chinese Exclusion Act, which was approved in 1882 and made Chinese residents the targets of the nation’s first law limiting immigration based on race or nationality.

The city had five Chinatowns but the largest one was built in 1872. Fifteen years later, the city council declared it a public nuisance and unanimously approved an order to remove it to make way for a new City Hall. Before officials acted, the thriving Chinatown was burned down by arsonists, destroying hundreds of homes and businesses and displacing about 1,400 people, according to the resolution.

“An apology for grievous injustices cannot erase the past, but admission of the historic wrongdoings committed can aid us in solving the critical problems of racial discrimination facing America today,” the resolution reads.
"""

# Split the text into sentences using SpaCy for more robust tokenization
doc = nlp(text)
sentences = [sent.text for sent in doc.sents]

# Find sentences with 'Chinese' or 'Asian' and their surrounding sentences
relevant_sentences = []
keywords = ['Chinese', 'Asian']

for i, sentence in enumerate(sentences):
    if any(keyword in sentence for keyword in keywords):
        # Check if there are sentences before and after
        if i > 0 and i < len(sentences) - 1:
            relevant_sentences.append(sentences[i-1])
            relevant_sentences.append(sentence)
            relevant_sentences.append(sentences[i+1])

# Join the relevant sentences into a single string
relevant_text = ' '.join(relevant_sentences)

# Perform sentiment analysis using TextBlob
blob = TextBlob(relevant_text)
sentiment_scores = blob.sentiment

# Print the relevant text and sentiment scores
print("Relevant Text:", relevant_text)
print("Sentiment Scores:", sentiment_scores)
