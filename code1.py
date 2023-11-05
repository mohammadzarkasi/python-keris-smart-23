import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import geneticalgorithm as ga

# Langkah 1: Siapkan dataset dan preprocess data
# Misalkan Anda telah mempersiapkan data dan menyimpannya sebagai numpy arrays dengan nama X dan y.
# X adalah array berisi gambar-gambar sampah dan y adalah array berisi label kelas (0 untuk anorganik, 1 untuk organik)

# Langkah 2: Buat model DenseNet121
def create_densenet_model():
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)  # Output layer dengan 1 neuron karena ini adalah klasifikasi biner
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

# Langkah 3: Optimasi hyperparameter menggunakan Genetic Algorithm
def fitness_function(params):
    learning_rate = params[0]
    num_epochs = int(params[1])
    batch_size = int(params[2])

    # Split data menjadi training dan validation set
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Buat model DenseNet121 baru dengan setiap iterasi untuk menerapkan hyperparameter yang diubah
    model = create_densenet_model()

    # Compile model dengan hyperparameter baru
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # Train model dengan hyperparameter baru
    model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs, validation_data=(X_val, y_val), verbose=0)

    # Evaluasi model pada validation set dan kembalikan nilai akurasi sebagai fitness
    _, accuracy = model.evaluate(X_val, y_val)
    return accuracy

parameter_ranges = {
    'learning_rate': (0.0001, 0.01),
    'num_epochs': (5, 30),
    'batch_size': (16, 128)
}

model_params = ga.geneticalgorithm(function=fitness_function, dimension=3, variable_type='real', variable_boundaries=[(parameter_ranges['learning_rate']), (parameter_ranges['num_epochs']), (parameter_ranges['batch_size'])], algorithm_parameters={'max_num_iteration': 10, 'population_size': 10, 'mutation_probability': 0.1, 'elit_ratio': 0.01, 'parents_portion': 0.3, 'crossover_probability': 0.5, 'crossover_type': 'uniform', 'max_iteration_without_improv': None})
best_params = model_params.output_dict['variable']
print("Best hyperparameters:", best_params)

# Gunakan hyperparameter terbaik untuk membuat dan melatih model akhir
best_learning_rate, best_num_epochs, best_batch_size = best_params
best_learning_rate = float(best_learning_rate)
best_num_epochs = int(best_num_epochs)
best_batch_size = int(best_batch_size)

# Split data menjadi training dan validation set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Buat model DenseNet121 dengan hyperparameter terbaik
best_model = create_densenet_model()

# Compile model dengan hyperparameter terbaik
optimizer = Adam(learning_rate=best_learning_rate)
best_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train model dengan hyperparameter terbaik
best_model.fit(X_train, y_train, batch_size=best_batch_size, epochs=best_num_epochs, validation_data=(X_val, y_val), verbose=1)

# Langkah 4: Klasifikasi sampah organik dan anorganik serta pemilihan sampah hijau dan coklat
def classify_waste(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Klasifikasi organik atau anorganik
    is_organic = best_model.predict(img_array)[0][0] >= 0.5

    if is_organic:
        # Klasifikasi sampah hijau atau coklat
        # Misalkan Anda memiliki model hijau_coklat_model yang telah dilatih sebelumnya
        # model hijau_coklat_model harus menerima gambar berukuran 224x224 dan mengeluarkan label 'hijau' atau 'coklat'
        label = hijau_coklat_model.predict(img_array)
        label = 'hijau' if label == 0 else 'coklat'
    else:
        label = 'anorganik'

    return label

# Contoh penggunaan
image_path = 'path_to_your_image.jpg'
result = classify_waste(image_path)
print("Hasil klasifikasi:", result)