
from src.divide_dataset import DatasetDivider
from src.visualize_dataset import DataVizualizer
from src.train import MLP
from src.predict import Predictor
from src.config import (
	MODEL_PATH,
	DATA_CSV_PATH,
	TRAIN_CSV_PATH,
	PREDICT_CSV_PATH,
	EPOCHS,
	LAYERS,
	DATA_SPLIT
)
import pandas as pd

def show_options():
	print("========= MENU =========")
	print("0. EXIT")
	print("1. DATA VIZUALIZER")
	print("2. MLP TRAIN")
	print("3. MLP PREDICT")
	print("Other. EXIT")
	print("========================")

if __name__ == '__main__':
	while (1):
		show_options()
		choice = input("Enter your choice: ")
		if choice.isdigit():
			choice = int(choice)
		if (choice == 1):
			visualizer = DataVizualizer(DATA_CSV_PATH)
			visualizer.get_infos()
			divider = DatasetDivider(DATA_CSV_PATH, DATA_SPLIT, TRAIN_CSV_PATH, PREDICT_CSV_PATH)
			divider.write_datasets()

		elif (choice == 2):
			mlp = MLP(TRAIN_CSV_PATH, LAYERS, EPOCHS)
			mlp.train()
		
		elif (choice == 3):
			predictor = Predictor(MODEL_PATH)
			predictions = predictor.predict_dataset(PREDICT_CSV_PATH)

		elif (choice == 0):
			break

		else: 
			break
