import urllib.request
from bs4 import BeautifulSoup
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from flask import Flask, request, render_template
from sklearn.model_selection import train_test_split

app = Flask(__name__)

def scrape_data(url):
    response = urllib.request.urlopen(url)
    web_content = response.read()
    soup = BeautifulSoup(web_content, 'html.parser')
    paragraphs = [para.get_text().strip() for para in soup.find_all('p') if para.get_text().strip() != '']
    return paragraphs

# URL for scraping data
data_url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
data = scrape_data(data_url)

# Assuming 'data' is already a list of text data as before
# Text vectorization
vectorizer = TextVectorization(max_tokens=10000, output_sequence_length=200)
vectorizer.adapt(data)

# Vectorize the data and convert to numpy array for compatibility
vectorized_data = vectorizer(data).numpy()

# Assuming labels are prepared correctly
labels = np.array([1 if 'artificial intelligence' in text.lower() else 0 for text in data])

# Split the data into training and validation sets
vectorized_data_train, vectorized_data_val, labels_train, labels_val = train_test_split(
    vectorized_data, labels, test_size=0.2, random_state=42)

app = Flask(__name__)

# Build a more complex model
model = Sequential([
    Embedding(input_dim=len(vectorizer.get_vocabulary()), output_dim=128, mask_zero=True),
    GlobalAveragePooling1D(),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# Train the model
model.fit(vectorized_data_train, labels_train, epochs=30, verbose=1, 
          validation_data=(vectorized_data_val, labels_val), callbacks=[early_stopping])

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        user_question = request.form['question']
        vectorized_question = vectorizer([user_question]).numpy()  # Convert question to numpy array
        prediction = model.predict(vectorized_question)[0]
        answer = "Related to AI." if prediction > 0.5 else "Not related to AI."
        return render_template('index.html', question=user_question, answer=answer, options=data[:10])
    return render_template('index.html', options=data[:10])

@app.route('/debug', methods=['POST'])
def debug():
    user_question = request.form['question']
    vectorized_question = vectorizer([user_question])
    prediction = model.predict(vectorized_question)[0]
    return f"Raw prediction value: {prediction}"

if __name__ == '__main__':
    app.run(debug=True)
