import seaborn as sns
import matplotlib.pyplot as plt
import csv 
import pandas as pd


class DataVizualizer:
	nb_lines = 0
	rows = []
	dataset_path = ""
	df = None

	def __init__(self, dataset_path):
		self.dataset_path = dataset_path

	def open_file(self):
		with open(self.dataset_path, newline='') as csvfile:
			reader = csv.reader(csvfile, delimiter=',')
			for row in reader:
				self.rows.append(row)
		self.df = pd.read_csv(self.dataset_path, header=None)
	
	def print_rows(self):
		for row in self.rows:
			print(row)

	def percentage(self):
		if (len(self.rows) == 0):
			self.open_file()
		nb_m = 0
		for row in self.rows:
			if (len(row)):
				if (row[1] == 'M'):
					nb_m += 1
			else:
				print("missing column in file ", self.dataset_path)
		percentage = nb_m * 100 / len(self.rows)
		print(f'{round(percentage, 2)}% of the sample in the dataset has a malignant cancer ({nb_m}/{len(self.rows)})')

	def get_infos(self):
		self.percentage()
		self.df.info()
		corr = self.df.corr(numeric_only=True)
		sns.heatmap(corr, cmap="coolwarm")
		plt.show()