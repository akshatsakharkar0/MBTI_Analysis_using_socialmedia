import pandas as pd
import numpy as np
import random
import os
import joblib
import torch
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from google.colab import drive
drive.mount('/content/drive')

# Create folders in Google Drive
base_dir = '/content/drive/MyDrive/personality_models'
os.makedirs(base_dir, exist_ok=True)

# Load dataset
dataset_path = "/content/drive/MyDrive/personality_models/mbti_dataset.csv"
df = pd.read_csv(dataset_path)

# Encode labels (EI dimension)
le_ei = LabelEncoder()
df['EI_encoded'] = le_ei.fit_transform(df['EI'])

X_train, X_val, y_train, y_val = train_test_split(df['body'], df['EI_encoded'], test_size=0.2, random_state=42)

# Set random seeds
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(42)

### ----------------- Classical ML Models -----------------

vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)

# 1. SVM
svm_model = LinearSVC(random_state=42, max_iter=10000)
svm_model.fit(X_train_tfidf, y_train)
y_pred_svm = svm_model.predict(X_val_tfidf)
acc_svm = accuracy_score(y_val, y_pred_svm)
print(f"SVM Accuracy: {acc_svm}")
joblib.dump(svm_model, os.path.join(base_dir, 'svm_model.pkl'))

import random
import numpy as np
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline


# Set random seeds for reproducibility
import torch
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# Assume 'df' is already loaded as a DataFrame with columns: 'body', 'EI', 'NS', 'FT', 'JP'
# Example loading (uncomment if needed):
# df = pd.read_csv('your_dataset.csv')


models_dir = os.path.join(base_dir, 'models')
os.makedirs(models_dir, exist_ok=True)

# Encode each MBTI dimension
label_encoders = {}
for dim in ['EI', 'NS', 'FT', 'JP']:
    le = LabelEncoder()
    df[f'{dim}_encoded'] = le.fit_transform(df[dim])
    label_encoders[dim] = le
    # Save the encoder for later use
    joblib.dump(le, os.path.join(models_dir, f'{dim}_label_encoder.pkl'))

# Prepare X and y
X = df['body']
y = df[['EI_encoded', 'NS_encoded', 'FT_encoded', 'JP_encoded']]

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the pipeline with MultiOutputClassifier
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(max_features=5000, stop_words='english')),
    ('classifier', MultiOutputClassifier(LogisticRegression(max_iter=1000, random_state=42)))
])

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate accuracy for each dimension
y_pred = pipeline.predict(X_val)
for idx, dim in enumerate(['EI', 'NS', 'FT', 'JP']):
    acc = accuracy_score(y_val.iloc[:, idx], y_pred[:, idx])
    print(f"{dim} Accuracy: {acc}")

# Save the entire pipeline
model_path = os.path.join(models_dir, 'pipeline.pkl')
joblib.dump(pipeline, model_path)

print("Model and encoders saved successfully!")


# 3. Random Forest
rf_model = RandomForestClassifier(n_estimators=80, max_depth=25, random_state=42)
rf_model.fit(X_train_tfidf, y_train)
y_pred_rf = rf_model.predict(X_val_tfidf)
acc_rf = accuracy_score(y_val, y_pred_rf)
print(f"Random Forest Accuracy: {acc_rf}")
joblib.dump(rf_model, os.path.join(base_dir, 'rf_model.pkl'))

### ----------------- Neural Network Models -----------------

# Tokenization for NN models
tokenizer_nn = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer_nn.fit_on_texts(X_train)
X_train_seq = tokenizer_nn.texts_to_sequences(X_train)
X_val_seq = tokenizer_nn.texts_to_sequences(X_val)

max_len = 200
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
X_val_pad = pad_sequences(X_val_seq, maxlen=max_len, padding='post', truncating='post')

num_classes = len(le_ei.classes_)
y_train_nn = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
y_val_nn = tf.keras.utils.to_categorical(y_val, num_classes=num_classes)



# 4. Dense NN
dense_model = Sequential([
    Dense(128, activation='relu', input_shape=(max_len,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])
dense_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
dense_model.fit(X_train_pad, y_train_nn, epochs=5, batch_size=32, validation_data=(X_val_pad, y_val_nn), verbose=0)
acc_dense = dense_model.evaluate(X_val_pad, y_val_nn, verbose=0)[1]
print(f"Dense NN Accuracy: {acc_dense}")
dense_model.save(os.path.join(base_dir, 'dense_model.keras'))


# 5. LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# Define the model
lstm_model = Sequential([
    Embedding(input_dim=5000, output_dim=64, input_length=max_len),
    LSTM(64, dropout=0.3),  # Removed recurrent_dropout for better GPU optimization
    Dense(num_classes, activation='softmax')
])

# Compile the model
lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks to monitor and save best model
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
    ModelCheckpoint(filepath=os.path.join(base_dir, 'lstm_model.keras'),
                    monitor='val_loss', save_best_only=True, verbose=1)
]

# Train the model with verbose output
history = lstm_model.fit(
    X_train_pad, y_train_nn,
    epochs=5,
    batch_size=32,
    validation_data=(X_val_pad, y_val_nn),
    verbose=1,  # Show progress updates
    callbacks=callbacks
)
acc_lstm = lstm_model.evaluate(X_val_pad, y_val_nn, verbose=1)[1] 
print(f"LSTM Accuracy: {acc_lstm}")
lstm_model.save(os.path.join(base_dir, 'lstm_model.keras'))

### ----------------- Transformer Model -----------------

# Using smaller sample for demonstration
sample_size = 500
X_train_sample = X_train[:sample_size]
y_train_sample = y_train[:sample_size]
X_val_sample = X_val[:sample_size]
y_val_sample = y_val[:sample_size]

tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer_bert(list(X_train_sample), truncation=True, padding=True, max_length=128)
val_encodings = tokenizer_bert(list(X_val_sample), truncation=True, padding=True, max_length=128)

class MBTIDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

train_dataset = MBTIDataset(train_encodings, y_train_sample.tolist())
val_dataset = MBTIDataset(val_encodings, y_val_sample.tolist())

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)
bert_model.to(device)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

optim = AdamW(bert_model.parameters(), lr=5e-5)

# Training
bert_model.train()
for batch in tqdm(train_loader, desc="Training BERT"):
    optim.zero_grad()
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    outputs = bert_model(input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward()
    optim.step()

# Evaluation
bert_model.eval()
preds = []
true = []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = bert_model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        preds.extend(predictions.cpu().numpy())
        true.extend(labels.cpu().numpy())
acc_bert = accuracy_score(true, preds)
print(f"BERT Accuracy: {acc_bert}")

# Save BERT model
bert_dir = os.path.join(base_dir, 'bert_model')
os.makedirs(bert_dir, exist_ok=True)
bert_model.save_pretrained(bert_dir)
tokenizer_bert.save_pretrained(bert_dir)

### ----------------- Final Report -----------------

results = {
    "Model": ["SVM", "Logistic Regression", "Random Forest", "Dense NN", "LSTM", "BERT"],
    "Accuracy": [acc_svm, acc_lr, acc_rf, acc_dense, acc_lstm, acc_bert]
}
results_df = pd.DataFrame(results)
print("\nFinal Results:\n", results_df)

print("\nAll models have been saved to Google Drive and loaded successfully!")