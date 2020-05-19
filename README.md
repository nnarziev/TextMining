# TextMining
A platform for text mining purpose to see the trends of words and different phrases across different years that has a web front end. 
The user can give and input as a text or any type of file with text and can specify the year of that input; a platform gets input and first, processes it by extracting text from the file and tokenizing it into words and phrases using NLTK library. 
After the tokenization process it stores all the words and phrases in the PostgreSQL database. 
On the website, the user can also visualize the words and phrases by applying different filters (e.g., year filter). 
Visualization is represented by charts, word cloud, and different topic clusters during the specific year so that we can monitor the trends of that year. Words and phrases are separated into clusters using k-means algorithm on embeddings that are taken using the BERT model.
