import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
import json


sign_dict = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E',
    5: 'F',
    6: 'G',
    7: 'H',
    8: 'I',
    9: 'J',
    10: 'K',
    11: 'L',
    12: 'M',
    13: 'N',
    14: 'O',
    15: 'P',
    16: 'Q',
    17: 'R',
    18: 'S',
    19: 'T',
    20: 'U',
    21: 'V',
    22: 'W',
    23: 'X',
    24: 'Y',
    25: 'Z',
}

class CNN:
    def __init__(self, dataset, optimizer, epochs=10, verbose=1):
        tf.random.set_seed(242)
        self.model = None
        self.model_optimizer = optimizer
        self.epochs = epochs
        self.batch_size = 32
        self.num_classes = 26
        self.input_shape = 56
        self.verbose = verbose
        self.x_train = dataset["x_train"]
        self.y_train = dataset["y_train"]
        self.x_test = dataset["x_test"]
        self.y_test = dataset["y_test"]
        self.x_val = dataset["x_val"]
        self.y_val = dataset["y_val"]
        self.history = None
        self.model_loss = 'categorical_crossentropy'
        self.model_metrics = ['accuracy', 'f1_score']
        self.model_layers = [
            tf.keras.layers.Input(shape=(self.input_shape, self.input_shape, 1)),
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(self.num_classes, activation='softmax'),
        ]

        self.get_data()
        self.compile_model()

    def get_data(self):

        self.x_train = np.array(self.x_train).reshape((self.x_train.shape[0], self.input_shape, self.input_shape, 1))
        one_hot_labels = np.zeros((self.y_train.shape[0], 26))
        one_hot_labels[np.arange(self.y_train.size), self.y_train.astype(int)] = 1
        self.y_train = one_hot_labels

        self.x_val = np.array(self.x_val).reshape((self.x_val.shape[0], self.input_shape, self.input_shape, 1))
        one_hot_labels = np.zeros((self.y_val.shape[0], 26))
        one_hot_labels[np.arange(self.y_val.size), self.y_val.astype(int)] = 1
        self.y_val = one_hot_labels

        self.x_test = np.array(self.x_test).reshape((self.x_test.shape[0], self.input_shape, self.input_shape, 1))
        one_hot_labels = np.zeros((self.y_test.shape[0], 26))
        one_hot_labels[np.arange(self.y_test.size), self.y_test.astype(int)] = 1
        self.y_test = one_hot_labels

        self.x_train = self.x_train / 255.0
        self.y_train = self.y_train
        self.x_val = self.x_val / 255.0
        self.y_val = self.y_val
        self.x_test = self.x_test / 255.0      
        self.y_test = self.y_test
        return

    def compile_model(self):
        self.model = tf.keras.Sequential(self.model_layers)
        self.model.compile(loss=self.model_loss, optimizer=self.model_optimizer, metrics=self.model_metrics)
        if self.verbose: self.model.summary()
        return

    def train(self):
        # print('Training...')
        self.history = self.model.fit(self.x_train, self.y_train, 
                                      batch_size=self.batch_size, epochs=self.epochs, 
                                      verbose=self.verbose, validation_data=(self.x_val, self.y_val),
                                      callbacks=[])
        return

    def evaluate(self):
        print('Evaluating...')
        self.model.evaluate(self.x_test, self.y_test, verbose=1)
        return
    
    def predict(self):
        probs = self.model.predict(self.x_test)
        pred = np.argmax(probs, axis=1)#.numpy()
        true = np.argmax(self.y_test, axis=1)#.numpy()

        return pred, true

    def save_history(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        ax1.plot(self.history.history['val_accuracy'], label='Val')
        ax1.plot(self.history.history['accuracy'], label='Train')
        ax1.set_title('Accuracy durante el entrenamiento con ' + self.model_optimizer.name)
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Época')
        ax1.grid()
        ax1.legend(loc='center right')
        ax1.set_ylim(0, 1.05)
        # ax1.set_xticks(np.arange(0, self.epochs, 2))

        ax2.plot(self.history.history['val_loss'], label='Val')
        ax2.plot(self.history.history['loss'], label='Train')
        ax2.set_title('Pérdida durante el entrenamiento con ' + self.model_optimizer.name)
        ax2.set_ylabel('Pérdida')
        ax2.set_xlabel('Época')
        ax2.legend(loc='upper right')
        ax2.grid()
        # ax2.set_xticks(np.arange(0, self.epochs, 2))

        plt.tight_layout()
        plt.savefig("output/{}_metrics.png".format(self.model_optimizer.name))
        plt.clf()

    def save_model(self):
        self.model.save("models/{}_model_big.keras".format(self.model_optimizer.name))
        return

    def load_model(self, model_name):
        self.model = tf.keras.models.load_model("models/{}_model.keras".format(model_name))
        return
        

def run_models(dataset, metric):
    optimizers = tf.keras.optimizers
    lr = 0.001
    beta_1 = 0.9
    beta_2 = 0.999
    print("Graficando metricas...")

    fig, axes = plt.subplots(nrows=6, ncols=4, figsize=(20, 20))
    optimizers_list = [
        lambda: optimizers.SGD(learning_rate=lr, name="DGE"), 
        lambda: optimizers.SGD(learning_rate=lr, momentum=beta_1, name="Momentum"),
        lambda: optimizers.RMSprop(learning_rate=lr, name="RMSprop"),
        lambda: optimizers.Adam(learning_rate=lr, beta_1=beta_1, beta_2=beta_2, name="Adam"),
        ]
    epochs = [5, 10, 20, 50, 75, 100]
    metric_values_1 = {
        "DGE": {},
        "Momentum": {},
        "RMSprop": {},
        "Adam": {},
    }
    for i, epoch in enumerate(epochs):  
        for j, optimizer_func in enumerate(optimizers_list):
            ax = axes[i, j]
            optimizer = optimizer_func()
            name = optimizer.name
            # Datos de ejemplo para cada subgráfico
            model = CNN(dataset, optimizer=optimizer, verbose=0, epochs=epoch)
            model.train()
            hist = model.history

            metric_values_1[optimizer.name][epoch] = {
                "train": hist.history[f'{metric}'][-1],
                "val": hist.history[f'val_{metric}'][-1]
            }
        

            # Grafico individual
            single_fig, single_ax = plt.subplots(figsize=(10, 10))
            single_ax.plot(range(1, epoch + 1),hist.history[f'val_{metric}'], label='Val')
            single_ax.plot(range(1, epoch + 1),hist.history[f'{metric}'], label='Train')
            single_ax.grid()
            if metric == "loss":
                single_ax.set_ylim(0, 4.05)
                single_ax.set_yticks(np.arange(0, 4.05, 1))
            else:
                single_ax.set_ylim(0, 1.05)
                single_ax.set_yticks(np.arange(0, 1.05, 0.25))
            single_ax.set_title(f"{name}: k = {epoch}, α = 0.001", fontdict={'fontsize': 16, "fontweight": "bold"})
            single_ax.set_ylabel(f"{metric.capitalize()}", fontdict={'fontsize': 14,})
            single_ax.set_xlabel(f"Épocas", fontdict={'fontsize': 14,})
            plt.savefig(f"output/indiv_charts/{name}_{metric}_{epoch}_1.png")
            plt.close(single_fig)

            # Grafico global
            ax.plot(range(1, epoch + 1),hist.history[f'val_{metric}'], label='Val')
            ax.plot(range(1, epoch + 1),hist.history[f'{metric}'], label='Train')
            ax.grid()
            ax.legend().set_visible(False)
            if metric == "loss":
                ax.set_ylim(0, 4.05)
                ax.set_yticks(np.arange(0, 4.05, 1))
            else:
                ax.set_ylim(0, 1.05)
                ax.set_yticks(np.arange(0, 1.05, 0.25))

            if i == 0:  # Títulos de las columnas (optimizadores)
                ax.set_title(optimizer.name, fontdict={'fontsize': 16, "fontweight": "bold"})
            if j == 0:  # Etiquetas de las filas (épocas)
                ax.set_ylabel(f"Épocas: {epoch}", fontdict={'fontsize': 14,})
    plt.tight_layout()
    plt.savefig(f"output/{metric}_metrics_1.png")
    plt.close(fig)

    fig, axes = plt.subplots(nrows=6, ncols=4, figsize=(20, 20))
    alphas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
    optimizers_dict = {
        "DGE": optimizers.SGD,
        "Momentum": optimizers.SGD,
        "RMSprop": optimizers.RMSprop,
        "Adam": optimizers.Adam,
    }
    epochs_dict = {
        "DGE": 75,
        "Momentum": 25,
        "RMSprop": 10,
        "Adam": 5,
    }

    metric_values_2 = {
        "DGE": {},
        "Momentum": {},
        "RMSprop": {},        
        "Adam": {},
    }
    for i, alpha in enumerate(alphas):
        for j, optimizer_info in enumerate(optimizers_dict.items()):
            # print(optimizer_info)
            ax = axes[i, j]
            name = optimizer_info[0]
            epochs = epochs_dict[name]
            if name == "DGE":
                optimizer = optimizer_info[1](learning_rate=alpha, name=name)
            elif name == "Momentum":
                optimizer = optimizer_info[1](learning_rate=alpha, momentum=beta_1, name=name)
            elif name == "Adam":
                optimizer = optimizer_info[1](learning_rate=alpha, beta_1=beta_1, beta_2=beta_2, name=name)
            elif name == "RMSprop":
                optimizer = optimizer_info[1](learning_rate=alpha, name=name)
            else:
                print("Error")
                return

            model = CNN(dataset, optimizer=optimizer, verbose=0, epochs=epochs)
            model.train()
            hist = model.history

            metric_values_2[optimizer.name][alpha] = {
                "train": hist.history[f'{metric}'][-1],
                "val": hist.history[f'val_{metric}'][-1]
            }
            

            # Grafico individual
            single_fig, single_ax = plt.subplots(figsize=(10, 10))
            single_ax.plot(range(1, epochs + 1), hist.history[f'val_{metric}'], label='Val')
            single_ax.plot(range(1, epochs + 1), hist.history[f'{metric}'], label='Train')
            single_ax.grid()
            if metric == "loss":
                single_ax.set_ylim(0, 4.05)
                single_ax.set_yticks(np.arange(0, 4.05, 1))
            else:
                single_ax.set_ylim(0, 1.05) 
                single_ax.set_yticks(np.arange(0, 1.05, 0.25))
            single_ax.set_title(f"{name}: k = {epochs}, α = {alpha}", fontdict={'fontsize': 16, "fontweight": "bold"})
            single_ax.set_ylabel(f"{metric.capitalize()}", fontdict={'fontsize': 14,})
            single_ax.set_xlabel(f"Épocas", fontdict={'fontsize': 14,})
            plt.savefig(f"output/indiv_charts/{name}_{metric}_{alpha}_2.png")
            plt.close(single_fig)

            # Grafico global
            ax.plot(range(1, epochs + 1), hist.history[f'val_{metric}'], label='Val')
            ax.plot(range(1, epochs + 1), hist.history[f'{metric}'], label='Train')
            ax.grid()
            ax.legend().set_visible(False)
            if metric == "loss":
                ax.set_ylim(0, 4.05)
                ax.set_yticks(np.arange(0, 4.05, 1))
            else:
                ax.set_ylim(0, 1.05)
                ax.set_yticks(np.arange(0, 1.05, 0.25))


            if i == 0:  # Títulos de las columnas (optimizadores)
                ax.set_title(f"{name}: {epochs} épocas", fontdict={'fontsize': 16, "fontweight": "bold"})
            if j == 0:  # Etiquetas de las filas (épocas)
                ax.set_ylabel(f"Alpha = {alpha}", fontdict={'fontsize': 14,})
    plt.tight_layout()
    plt.savefig(f"output/{metric}_metrics_2.png")
    plt.close(fig)

    with open(f"output/{metric}_metrics_1.json", "w") as f:
        json.dump(metric_values_1, f)
        f.close()
    with open(f"output/{metric}_metrics_2.json", "w") as f:
        json.dump(metric_values_2, f)
        f.close()

def create_model(dataset):
    model = CNN(dataset, optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, name='final'), 
                verbose=0, epochs=10)
    model.train()
    pred, true = model.predict()
    model.save_model()

    print(classification_report(true, pred))

    labels = list(sign_dict.values())
    del labels[9] # J
    del labels[24] # Z

    cm = confusion_matrix(true, pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cbar=False)
    plt.xlabel("Etiquetas predichas")
    plt.ylabel("Etiquetas verdaderas")
    plt.title("Matriz de confusión")
    plt.savefig("output/model_confusion_matrix.png")
    plt.show()
    plt.clf()

    return

def final_models():
    np.random.seed(21)
    tf.random.set_seed(21)
    dataset = create_dataset()
    dge = CNN(dataset, optimizer=tf.keras.optimizers.SGD(learning_rate=0.05, name='DGE_final'), verbose=0, epochs=10)
    momentum = CNN(dataset, optimizer=tf.keras.optimizers.SGD(learning_rate=0.02, momentum=0.9, name='Momentum_final'), verbose=0, epochs=10)
    rmsprop = CNN(dataset, optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.002, name='RMSprop_final'), verbose=0, epochs=10)
    adam = CNN(dataset, optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, name='Adam_final'), verbose=0, epochs=10)

    print("Entrenando SGD...")
    dge.train()
    print("Entrenando Momentum...")
    momentum.train()
    print("Entrenando RMSprop...")
    rmsprop.train()
    print("Entrenando Adam...")
    adam.train()

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))
    ax = axes[0, 0]
    ax.plot(range(1, 11), dge.history.history['accuracy'], label='DGE')
    ax.plot(range(1, 11), momentum.history.history['accuracy'], label='Momentum')
    ax.plot(range(1, 11), rmsprop.history.history['accuracy'], label='RMSprop')
    ax.plot(range(1, 11), adam.history.history['accuracy'], label='Adam')
    ax.grid()
    ax.legend(fontsize=20)
    ax.set_title("Acc. conjunto de entrenamiento", fontdict={'fontsize': 20, "fontweight": "bold"})
    ax.set_ylabel("Acc.")
    ax.set_xlabel("Épocas")

    ax = axes[0, 1]
    ax.plot(range(1, 11), dge.history.history['val_accuracy'], label='DGE')
    ax.plot(range(1, 11), momentum.history.history['val_accuracy'], label='Momentum')
    ax.plot(range(1, 11), rmsprop.history.history['val_accuracy'], label='RMSprop')
    ax.plot(range(1, 11), adam.history.history['val_accuracy'], label='Adam')
    ax.grid()
    ax.legend(fontsize=20)
    ax.set_title("Acc. conjunto de validación", fontdict={'fontsize': 20, "fontweight": "bold"})
    ax.set_ylabel("Acc.")
    ax.set_xlabel("Épocas")

    ax = axes[1, 0]
    ax.plot(range(1, 11), dge.history.history['loss'], label='DGE')
    ax.plot(range(1, 11), momentum.history.history['loss'], label='Momentum')
    ax.plot(range(1, 11), rmsprop.history.history['loss'], label='RMSprop')
    ax.plot(range(1, 11), adam.history.history['loss'], label='Adam')
    ax.grid()
    ax.legend(fontsize=20)
    ax.set_title("Pérdida conjunto de entrenamiento", fontdict={'fontsize': 20, "fontweight": "bold"})
    ax.set_ylabel("Pérdida")
    ax.set_xlabel("Épocas")

    ax = axes[1, 1]
    ax.plot(range(1, 11), dge.history.history['val_loss'], label='DGE')
    ax.plot(range(1, 11), momentum.history.history['val_loss'], label='Momentum')
    ax.plot(range(1, 11), rmsprop.history.history['val_loss'], label='RMSprop')
    ax.plot(range(1, 11), adam.history.history['val_loss'], label='Adam')
    ax.grid()
    ax.legend(fontsize=20)
    ax.set_title("Pérdida conjunto de validación", fontdict={'fontsize': 20, "fontweight": "bold"})
    ax.set_ylabel("Pérdida")
    ax.set_xlabel("Épocas")

    plt.tight_layout()
    plt.savefig("output/final_models.png")
    # plt.show()
    plt.close(fig)
    metrics = {
        "train_acc": {
            "dge": dge.history.history['accuracy'][-1],
            "momentum": momentum.history.history['accuracy'][-1],
            "rmsprop": rmsprop.history.history['accuracy'][-1],
            "adam": adam.history.history['accuracy'][-1],
        },
        "val_acc": {
            "dge": dge.history.history['val_accuracy'][-1],
            "momentum": momentum.history.history['val_accuracy'][-1],
            "rmsprop": rmsprop.history.history['val_accuracy'][-1],
            "adam": adam.history.history['val_accuracy'][-1],
        },
        "train_loss": {
            "dge": dge.history.history['loss'][-1],
            "momentum": momentum.history.history['loss'][-1],
            "rmsprop": rmsprop.history.history['loss'][-1],
            "adam": adam.history.history['loss'][-1],
        },
        "val_loss": {
            "dge": dge.history.history['val_loss'][-1],
            "momentum": momentum.history.history['val_loss'][-1],
            "rmsprop": rmsprop.history.history['val_loss'][-1],
            "adam": adam.history.history['val_loss'][-1],
        }
    }

    with open(f"output/final_metrics.json", "w") as f:
        json.dump(metrics, f)
        f.close()

def create_dataset():
    df_train = pd.read_csv("data/sign_mnist.csv")
    X = df_train.drop(columns=['label'])
    y = df_train['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=242)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=242)

    dataset = {
        "x_train": X_train,
        "y_train": y_train,
        "x_val": X_val,
        "y_val": y_val,
        "x_test": X_test,    
        "y_test": y_test
    }
    return dataset

if __name__ == "__main__":
    # dataset = create_dataset()
    # run_models(dataset, "accuracy")
    # run_models(dataset, "loss")
    # create_model(dataset)

    final_models()


