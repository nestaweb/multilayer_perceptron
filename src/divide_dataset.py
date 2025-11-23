import csv
import pandas as pd

class DatasetDivider:
	df = None
	rows = []
	training_rows = []
	predicting_rows = []

	def __init__(self, dataset_path = "", training_percentage = 0.7, training_file_path = "../dataset/training.csv", predict_file_path = "../dataset/predict.csv"):
		self.dataset_path = dataset_path
		self.training_file_path = training_file_path
		self.predict_file_path = predict_file_path
		self.training_percentage = training_percentage

	def open_file(self):
		print(self.dataset_path)
		with open(self.dataset_path, newline='') as csvfile:
			reader = csv.reader(csvfile, delimiter=',')
			for row in reader:
				self.rows.append(row)
		self.df = pd.read_csv(self.dataset_path, header=None)
	
	def split_dataset(self):
		self.open_file()
		stop_num = round(len(self.rows) * self.training_percentage)
		self.training_rows += self.rows[0:stop_num]
		self.predicting_rows += self.rows[stop_num+1:]
		print(f'{len(self.training_rows)} Training samples created')
		print(f'{len(self.predicting_rows)} Predicting samples created')

	def write_datasets(self):
		self.split_dataset()
		with open(self.training_file_path, 'w', newline='') as csvfile:
			writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
			writer.writerows(self.training_rows)

		with open(self.predict_file_path, 'w', newline='') as csvfile:
			writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
			writer.writerows(self.predicting_rows)