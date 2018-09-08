'''
Ref:
https://www.datacamp.com/community/tutorials/recommender-systems-python

'''
import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('stopwords')

class RecommendationMovies:
	def __init__(self, dataset):
		self.data = dataset.iloc[1:5]
		self.num_votes = 0
		self.mean_vote = 0

	def simple_rec(self, percentile):
		'''
		Decription: Recommends the top items based on a certain metric
			or score. Weighted rating based on vote count and vote average.
		@param "percentile": Top percentile of movies to be considered
		@return "simple_data": DataFrame arranged in descending order of scores
		'''
		self.num_votes = data['vote_count'].quantile(percentile)
		self.mean_vote = data['vote_average'].mean()
		
		# Find labels where vote count is > num_votes
		simple_data = data.copy().loc[data['vote_count'] >= self.num_votes]
		
		# Apply the formula for average movie ratings
		simple_data['score'] = simple_data.apply(self.rating, axis=1)

		# Sort movies based on the score
		simple_data = simple_data.sort_values(by=['score'], ascending=False)
		return simple_data

	def rating(self, row):
		tmp3 = row['vote_count'] + self.num_votes
		tmp1 = row['vote_count'] * row['vote_average'] / tmp3
		tmp2 = self.num_votes * self.mean_vote / tmp3
		return tmp1 + tmp2

	def content_rec(self):
		# Create document vs word matrix
		# TF = element/sum_row
		# IDF = log_e(sum_col / element)
		# TF-IDF weight = (word * 1 matrix) = 
		# print data['overview'].head(10)
		tf_idf = pd.DataFrame(index=self.data.index.values.tolist())
		
		c = 0
		for index in self.data.index.values.tolist():
			list1 = []
			list1 = self.data.loc[index,'overview']
			list1 = [re.sub(pattern=r'[\!\"#$%&\*\'+,-.\\\/:;<=>?@^_`()|~=]',
                            repl='',
                            string=list1)]
			list1 = list(filter(None, list1[0].split(' ')))
			print list1
			s1 = pd.Series(np.zeros(len(list1),1))
			for word in list1:
			# 	if word in d.keys():
			# 		d[word] += 1
			# 	else:
			# 		d[word] = 1
			# l = []
			# l.append(d)
				if word in tf_idf.columns.values.tolist():
					# tf_idf.loc[index, word] += 1
					s1[word] += 1
				else: 
					s1[word]  =1
					# tf_idf.loc[index, word] = 1
			tf_idf = tf_idf.append(s1, ignore_index=True)
		print tf_idf
			# tf_idf = pd.DataFrame(l)
			# print tf_idf

			


	def content_rec1(self):
		tfidf = TfidfVectorizer(stop_words='english')
		data['overview'] = data['overview'].fillna('')
		tfidf_matrix = tfidf.fit_transform(data['overview'])
		print tfidf_matrix.shape

if __name__ == "__main__":
	print '-------------------------------------'
	filename = "/home/shruti/Documents/code_sk/Datasets/the-movies-dataset/movies_metadata.csv"
	data = pd.read_csv(filename)

	rec = RecommendationMovies(data)

	# Building a simple recommender
	simple_data = rec.simple_rec(0.9)
	# print 'Simple recommendation:'
	# print simple_data[['title', 'vote_count', 'score']].head(15)

	# Building a content based recommender
	content_data1 = rec.content_rec1()

	content_data = rec.content_rec()

