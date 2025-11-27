import numpy as np
import pandas as pd
import sys
import os.path
from .config import DATA_CSV_PATH, PREDICT_CSV_PATH, TRAIN_CSV_PATH
from .divide_dataset import DatasetDivider

class Predictor:
	def __init__(self, model_path):
		self.model_path = model_path
		self.layers_weights = []
		self.layers_biases = []
		self.load_model()
	
	def load_model(self):
		try:
			model_data = np.load(self.model_path, allow_pickle=True).item()
			self.layers_weights = model_data['weights']
			self.layers_biases = model_data['biases']
			print(f"✅ Model loaded from '{self.model_path}'")
			print(f"   Architecture: {[w.shape[0] for w in self.layers_weights]} -> {self.layers_weights[-1].shape[1]}")
		except Exception as e:
			print(f"❌ Error loading model: {e}")
			sys.exit(1)
	
	def sigmoid(self, x):
		x = np.clip(x, -500, 500)
		return 1 / (1 + np.exp(-x))
	
	def softmax(self, x):
		e_x = np.exp(x - np.max(x))
		return e_x / e_x.sum()
	
	def forward(self, input_data):
		activation = input_data
		
		for i in range(len(self.layers_weights) - 1):
			z = (activation @ self.layers_weights[i]) + self.layers_biases[i]
			activation = self.sigmoid(z)
		
		z = (activation @ self.layers_weights[-1]) + self.layers_biases[-1]
		activation = self.softmax(z)
		
		return activation
	
	def predict_one(self, input_data):
		output = self.forward(input_data)
		predicted_class = np.argmax(output)
		confidence = output[predicted_class]
		label = 'B' if predicted_class == 0 else 'M'
		return label, confidence, output
	
	def predict_dataset(self, dataset_path):
		print(f"\n📊 Loading dataset from '{dataset_path}'...")
		if (not os.path.isfile(dataset_path)):
			print("dataset file for predicting creating it now ...")
			divider = DatasetDivider(DATA_CSV_PATH, 0.7, TRAIN_CSV_PATH, PREDICT_CSV_PATH)
			divider.write_datasets()
		df = pd.read_csv(dataset_path)
		float_df = df.select_dtypes(include=['float64'])
		
		has_labels = 'diagnosis' in df.columns or len(df.select_dtypes(include=['object']).columns) > 0
		
		if has_labels:
			object_df = df.select_dtypes(include=['object'])
			true_labels = object_df.values.ravel()
		
		data = float_df.values
		data = data / data.max(axis=0)
		
		print(f"   {len(data)} samples loaded\n")
		
		predictions = []
		correct = 0
		
		print("=" * 80)
		print(f"{'Index':<8} {'True':<8} {'Predicted':<12} {'Confidence':<12} {'B prob':<12} {'M prob':<12} {'Status'}")
		print("=" * 80)
		
		for i, row_data in enumerate(data):
			predicted_label, confidence, probs = self.predict_one(row_data)
			predictions.append(predicted_label)
			
			if has_labels:
				true_label = true_labels[i]
				is_correct = (predicted_label == true_label)
				if is_correct:
					correct += 1
				status = "✅" if is_correct else "❌"
				
				print(f"{i:<8} {true_label:<8} {predicted_label:<12} {confidence:.4f}      {probs[0]:.4f}      {probs[1]:.4f}      {status}")
			else:
				print(f"{i:<8} {'N/A':<8} {predicted_label:<12} {confidence:.4f}      {probs[0]:.4f}      {probs[1]:.4f}      -")
		
		print("=" * 80)
		
		if has_labels:
			accuracy = (correct / len(data)) * 100
			print(f"\n📈 Results:")
			print(f"   Total samples: {len(data)}")
			print(f"   Correct predictions: {correct}")
			print(f"   Wrong predictions: {len(data) - correct}")
			print(f"   Accuracy: {accuracy:.2f}%")
			
			b_count = predictions.count('B')
			m_count = predictions.count('M')
			print(f"\n   Predicted B (Benign): {b_count}")
			print(f"   Predicted M (Malignant): {m_count}")
		else:
			print(f"\n📈 Predictions completed:")
			print(f"   Total samples: {len(data)}")
			b_count = predictions.count('B')
			m_count = predictions.count('M')
			print(f"   Predicted B (Benign): {b_count}")
			print(f"   Predicted M (Malignant): {m_count}")
		
		return predictions