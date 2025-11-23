
from src.divide_dataset import DatasetDivider
from src.visualize_dataset import DataVizualizer
from src.train import MLP
from src.predict import Predictor
import pandas as pd

if __name__ == '__main__':
	model_path = "mlp.npy"
	data_csv_path = "dataset/data.csv"
	train_csv_path = "dataset/training.csv"
	predict_csv_path = "dataset/training.csv"

	while (1):
		print("Enter your choice:")
		choice = input()
		if choice.isdigit():
			choice = int(choice)
		if (choice == 1):
			visualizer = DataVizualizer(data_csv_path)
			visualizer.get_infos()
			divider = DatasetDivider(data_csv_path, 0.7, train_csv_path, predict_csv_path)
			divider.write_datasets()

		elif (choice == 2):
			mlp = MLP(train_csv_path, [30, 40, 32, 2], 200)
			mlp.open_dataset()
			mlp.train()
		
		elif (choice == 3):
			predictor = Predictor(model_path)
			predictions = predictor.predict_dataset(predict_csv_path)

		elif (choice == 0):
			break

		else: 
			break
