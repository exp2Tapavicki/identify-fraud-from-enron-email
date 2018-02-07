import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords')
sw = stopwords.words('english')
print len(sw)
print sw


from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
print(stemmer.stem("running"))