from os import path
from scipy.misc import imread
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy
import matplotlib.pyplot as plt
from PIL import Image

# Read the whole text.
df = pd.read_csv("train_set.csv",sep="\t")
my_category = df['Category']
my_content  = df['Content']

# read the mask image
# taken from
# http://rtyuiope.deviantart.com/art/Code-Geass-Wallpaper-374008098
zero_mask = numpy.array( Image.open( "zero.png" ) )
STOPWORDS.add("said")
wc = WordCloud(background_color="red", max_words=2000, mask=zero_mask, stopwords=STOPWORDS)

# generate word cloud
text = ""
for b in range(len(my_category.index)):
    if (my_category[b] == "Football"):
        text += my_content[b]
        

wc.generate(text)

# store to file
wc.to_file("football_cloud.png")

# show
plt.imshow(wc)
plt.axis("off")
plt.figure()
plt.imshow(zero_mask, cmap=plt.cm.gray)
plt.axis("off")
plt.show()