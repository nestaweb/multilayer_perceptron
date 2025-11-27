import random
import numpy as np
import pandas as pd
import os.path
from .config import DATA_CSV_PATH, PREDICT_CSV_PATH, TRAIN_CSV_PATH, LEARNING_RATE, PATIENCE, BATCH_SIZE, DATA_SPLIT, MOMENTUM, USE_NESTEROV
from .divide_dataset import DatasetDivider

class Layer():
	def __init__(self, input_size, output_size, activation="sigmoid", learning_rate=0.05, momentum=0.0, use_nesterov=False):
		self.activation = activation
		self.learning_rate = learning_rate
		self.momentum = momentum
		self.use_nesterov = use_nesterov
		self.W = np.random.uniform(-0.5, 0.5, (input_size, output_size))
        
		self.b = np.random.uniform(-0.1, 0.1, (output_size,))
        
		self.z = None
		self.a = None
		self.vW = np.zeros_like(self.W)
		self.vb = np.zeros_like(self.b)

	def sigmoid(self, x):
		x = np.clip(x, -500, 500)
		return 1 / (1 + np.exp(-x))

	def forward(self, input_activation):
		self.z = (input_activation @ self.W) + self.b
		self.a_prev = input_activation
		if (self.activation == "softmax"):
			self.a = self.softmax(self.z)
		else:
			self.a = self.sigmoid(self.z)

		return self.a

	def softmax(self, x):
		e_x = np.exp(x - np.max(x))
		return e_x / e_x.sum()

	def backprop(self, delta):
		if self.activation == "sigmoid":
			delta = delta * (self.a * (1 - self.a))
		# Reshape for matrix mult to avoid errors of different shape genre 16 != 2
		delta = delta.reshape(-1, 1)
		a_prev = self.a_prev.reshape(-1, 1)
		
		self.dW = a_prev @ delta.T
		self.db = delta.ravel()
		delta_prev = self.W @ delta 
		return delta_prev.ravel()

	def update_weights(self):
		if self.momentum <= 0:
			self.W -= self.learning_rate * self.dW
			self.b -= self.learning_rate * self.db
			return

		prev_vW = self.vW.copy()
		prev_vb = self.vb.copy()

		self.vW = self.momentum * self.vW - self.learning_rate * self.dW
		self.vb = self.momentum * self.vb - self.learning_rate * self.db

		if self.use_nesterov:
			self.W += -self.momentum * prev_vW + (1 + self.momentum) * self.vW
			self.b += -self.momentum * prev_vb + (1 + self.momentum) * self.vb
		else:
			self.W += self.vW
			self.b += self.vb

class MLP():
	def __init__(self, dataset_path, layers_sizes, epochs):
		self.dataset_path = dataset_path
		self.layers = []
		self.data = []
		self.epochs = epochs
		self.metrics = {
			"loss": [],
			"val_loss": [],
			"acc": [],
			"val_acc": []
		}
		self.init_network(layers_sizes)

	def open_dataset(self, validation_split=0.2):
		if (not os.path.isfile(self.dataset_path)):
			print("dataset file for predicting creating it now ...")
			divider = DatasetDivider(DATA_CSV_PATH, 0.7, TRAIN_CSV_PATH, PREDICT_CSV_PATH)
			divider.write_datasets()

		df = pd.read_csv(self.dataset_path)
		float_df = df.select_dtypes(include=['float64'])
		object_df = df.select_dtypes(include=['object'])
		raw_result = object_df.values.ravel()
		raw_data = float_df.values.tolist()
		
		data = np.array(raw_data)
		data = data / data.max(axis=0)
		
		split_idx = int(len(data) * (1 - validation_split))
		self.train_data = data[:split_idx]
		self.train_targets = raw_result[:split_idx]
		self.val_data = data[split_idx:]
		self.val_targets = raw_result[split_idx:]

	def init_network(self, layers_sizes):
		for i in range(len(layers_sizes) - 1):
			input_size = layers_sizes[i]
			output_size = layers_sizes[i+1]
			activation = "sigmoid"
			self.learning_rate = LEARNING_RATE
			if (len(layers_sizes) - 2 == i):
				activation = "softmax"
			layer_momentum = MOMENTUM if USE_NESTEROV else 0.0
			self.layers.append(Layer(input_size, output_size, activation, self.learning_rate, layer_momentum, USE_NESTEROV))

	def feedforward(self, row_data):
		activation = row_data
		for layer in self.layers:
			activation = layer.forward(activation)
		return activation

	def backprop(self, target):
		if target == 'B':
			target_vector = np.array([1.0, 0.0])
		elif target == 'M':
			target_vector = np.array([0.0, 1.0])
		else:
			raise ValueError(f"Unknown target label: {target}")
		delta = self.layers[-1].a - target_vector
		for layer in reversed(self.layers):
			delta = layer.backprop(delta)

	def compute_loss(self, data, targets):
		total_loss = 0
		for i, row_data in enumerate(data):
			prediction = self.feedforward(row_data)
			target = targets[i]
			
			if target == 'B':
				target_vector = np.array([1.0, 0.0])
			else:
				target_vector = np.array([0.0, 1.0])
			
			# Cross entropy ici
			total_loss += -np.sum(target_vector * np.log(prediction + 1e-10))
		
		return total_loss / len(data)

	def evaluate(self, data, targets):
		correct = 0
		for i, row_data in enumerate(data):
			prediction = self.feedforward(row_data)
			predicted_class = np.argmax(prediction)
			
			true_class = 0 if targets[i] == 'B' else 1
			
			if predicted_class == true_class:
				correct += 1
		
		accuracy = correct / len(data) * 100
		return accuracy
	
	def show_loss(self, epoch):
		loss = self.compute_loss(self.train_data, self.train_targets)
		val_loss = self.compute_loss(self.val_data, self.val_targets)
		train_acc = self.evaluate(self.train_data, self.train_targets)
		val_acc = self.evaluate(self.val_data, self.val_targets)
		
		self.metrics["loss"].append(loss)
		self.metrics["val_loss"].append(val_loss)
		self.metrics["acc"].append(train_acc)
		self.metrics["val_acc"].append(val_acc)
		print(f'epoch {epoch}/{self.epochs} - loss: {loss:.4f} - val_loss: {val_loss:.4f} - acc: {train_acc:.2f}% - val_acc: {val_acc:.2f}%')

	def show_curves(self):
		import matplotlib.pyplot as plt

		epochs = list(range(1, len(self.metrics["loss"]) + 1))

		plt.figure(figsize=(12, 5))

		plt.subplot(1, 2, 1)
		plt.plot(epochs, self.metrics["loss"], label='Loss')
		plt.plot(epochs, self.metrics["val_loss"], label='Val Loss')
		plt.xlabel("Epoch")
		plt.ylabel("Loss")
		plt.title("Training and Validation Loss")
		plt.legend()

		plt.subplot(1, 2, 2)
		plt.plot(epochs, self.metrics["acc"], label='Accuracy')
		plt.plot(epochs, self.metrics["val_acc"], label='Val Accuracy')
		plt.xlabel("Epoch")
		plt.ylabel("Accuracy (%)")
		plt.title("Training and Validation Accuracy")
		plt.legend()

		plt.tight_layout()
		plt.show()

	def train(self):
		current_lr = self.learning_rate
		best_val_loss = float('inf')
		patience = PATIENCE
		patience_counter = 0
		self.open_dataset()
		batch_size = min(BATCH_SIZE, len(self.train_data))
		min_epoch = self.epochs / 10
		
		for epoch in range(1, self.epochs + 1):
			indices = np.random.permutation(len(self.train_data))
			shuffled_data = self.train_data[indices]
			shuffled_targets = self.train_targets[indices]

			for batch_start in range(0, len(shuffled_data), batch_size):
				batch_data = shuffled_data[batch_start:batch_start + batch_size]
				batch_targets = shuffled_targets[batch_start:batch_start + batch_size]

				for layer in self.layers:
					layer.dW_acc = np.zeros_like(layer.W)
					layer.db_acc = np.zeros_like(layer.b)

				for row_data, target in zip(batch_data, batch_targets):
					self.feedforward(row_data)
					self.backprop(target)
					
					for layer in self.layers:
						layer.dW_acc += layer.dW
						layer.db_acc += layer.db

				for layer in self.layers:
					layer.learning_rate = current_lr
					layer.dW = layer.dW_acc / len(batch_data)
					layer.db = layer.db_acc / len(batch_data)
					layer.update_weights()
				
			val_loss = self.compute_loss(self.val_data, self.val_targets)
			if val_loss < best_val_loss:
				best_val_loss = val_loss
				patience_counter = 0
				best_weights = [layer.W.copy() for layer in self.layers]
				best_biases = [layer.b.copy() for layer in self.layers]
			elif (epoch >= min_epoch):
				patience_counter += 1
				if patience_counter >= patience:
					print(f"\n🛑 Early stopping at epoch {epoch}")
					for i, layer in enumerate(self.layers):
						layer.W = best_weights[i]
						layer.b = best_biases[i]
					break

			self.show_loss(epoch)
	
		model_data = {
			"weights": [layer.W for layer in self.layers],
			"biases":  [layer.b for layer in self.layers]
		}

		print("> saving model './mlp.npy' to disk...")
		np.save("mlp.npy", model_data, allow_pickle=True)
		print("> model saved at './mlp.npy'")

		while (1):
			choice = input("Enter epoch number to view details, Enter to exit or c for curves: ")
			if choice.isdigit():
				choice = int(choice) - 1
				if (choice >= 0 and choice <= self.epochs and choice <= len(self.metrics["loss"]) - 1):
					print(f'epoch {choice + 1}/{self.epochs} - loss: {self.metrics["loss"][choice]:.4f} - val_loss: {self.metrics["val_loss"][choice]:.4f} - acc: {self.metrics["acc"][choice]:.2f}% - val_acc: {self.metrics["val_acc"][choice]:.2f}%')
				else: 
					print("This epoch doesnt exist")
			elif (choice == "c"):
				self.show_curves()
			else: 
				print("\n")
				break