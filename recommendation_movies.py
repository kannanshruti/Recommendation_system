'''
Ref:
https://www.datacamp.com/community/tutorials/recommender-systems-python

'''
import pandas as pandas
from sklearn.feature_extraction.text import TfidfVectorizer

class RecommendationMovies:
	def __init__(self, dataset):
		self.data = dataset
		self.num_votes = 0
		self.mean_vote = 0

	def simple_rec(self, percentile):
		'''
		Decription: Recommends the top items based on a certain metric
			or score.
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
		

if __name__ == "__main__":
	filename = "/home/shruti/Documents/code_sk/Datasets/the-movies-dataset/movies_metadata.csv"
	data = pandas.read_csv(filename)

	rec = RecommendationMovies(data)

	# Building a simple recommender
	simple_data = rec.simple_rec(0.9)
	print 'Simple recommendation:'
	print simple_data[['title', 'vote_count', 'score']].head(15)
