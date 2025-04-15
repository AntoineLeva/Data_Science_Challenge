import h5py
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def explore_h5_data(filepath):
    """
    Cette fonction charge et affiche un aperçu des données à partir du fichier H5.
    """
    with h5py.File(filepath, 'r') as f:
        my_data_group = f['my_data']
        data = my_data_group[:]  # Lire toutes les données
        print(f"\nAperçu des données dans {filepath}:")
        print(data[:5])  # Affiche les 5 premières entrées pour un aperçu
    return data


# Fonction de normalisation des données MoCap
def normalize_mocap(data):
    nb_points = data.shape[1] // 3  # Divise en 3 pour chaque point
    data_reshaped = data.reshape(-1, nb_points, 3)
    ref = data_reshaped[:, 0:1, :]  # Point de référence (Premier point)
    data_centered = data_reshaped - ref
    return data_centered.reshape(data.shape[0], -1)


# Fonction de normalisation des données des capteurs de pression (Insoles)
def normalize_insoles(data):
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    return (data - min_vals) / (max_vals - min_vals + 1e-8)


# Chargement des données et normalisation
train_mocap = explore_h5_data("train_mocap.h5")
train_labels = explore_h5_data("train_labels.h5")
train_insoles = explore_h5_data("train_insoles.h5")
test_mocap = explore_h5_data("test_mocap.h5")
test_insoles = explore_h5_data("test_insoles.h5")

# Appliquer la normalisation
mocap_train_norm = normalize_mocap(train_mocap)
insoles_train_norm = normalize_insoles(train_insoles)
mocap_test_norm = normalize_mocap(test_mocap)
insoles_test_norm = normalize_insoles(test_insoles)

# Fusionner MoCap + Insoles pour chaque frame
X_train = np.hstack([mocap_train_norm, insoles_train_norm])
X_test = np.hstack([mocap_test_norm, insoles_test_norm])

# Créer les DataFrames finaux
df_train = pd.DataFrame(X_train)
df_train['label'] = train_labels.flatten()

df_test = pd.DataFrame(X_test)  # Pas de label ici

# Séparer les features (X) et les labels (y)
X = df_train.drop('label', axis=1)
y = df_train['label']

# Imputer les valeurs manquantes avec la moyenne des colonnes
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)  # On applique l'imputation

# Séparer les données en training et validation
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42, stratify=y
)


# X_train_split et y_train_split seront utilisés pour l'entraînement
# X_val_split et y_val_split pour la validation

def create_neural_network(input_dim):
    """
    Crée et retourne un modèle de réseau de neurones simple.
    """
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Si c'est une classification binaire

    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Exemple d'utilisation pour créer le modèle avec les dimensions de X_train_split
model = create_neural_network(X_train_split.shape[1])


def train_and_evaluate(X_train, X_val, y_train, y_val):
    """
    Entraîne un modèle, fait des prédictions et affiche les résultats de l'évaluation.
    """
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Prédictions sur les données de validation
    y_pred = clf.predict(X_val)

    # Évaluation du modèle
    print("Accuracy:", accuracy_score(y_val, y_pred))
    print(classification_report(y_val, y_pred))

    return clf


# Appliquer la fonction d'entraînement et évaluation
clf_trained = train_and_evaluate(X_train_split, X_val_split, y_train_split, y_val_split)


def predict_and_save_submission(model, df_test, imputer, output_path):
    """
    Effectue des prédictions sur les données de test et sauvegarde le fichier CSV de soumission.
    """
    test_imputed = imputer.transform(df_test)  # Imputer les données de test de la même façon
    test_preds = model.predict(test_imputed)

    # Préparer le DataFrame de soumission avec un ID (commençant à 1)
    submission = pd.DataFrame({
        'ID': np.arange(1, len(test_preds) + 1),  # Ajouter un ID qui commence à 1
        'Label': test_preds.flatten()
    })

    # Sauvegarder la soumission
    output_path = "submission.csv"  # Sauvegarde dans le répertoire courant pour tests locaux
    submission.to_csv(output_path, index=False)
    print(f"Fichier de soumission sauvegardé à {output_path}")


# Prédictions sur les données de test et sauvegarde
predict_and_save_submission(clf_trained, df_test, imputer, 'submission.csv')
