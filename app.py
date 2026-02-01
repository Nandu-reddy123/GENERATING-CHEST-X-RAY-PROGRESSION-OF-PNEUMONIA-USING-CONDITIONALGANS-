from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tfp
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.secret_key = 'abcd123'

model = load_model('MobileNetV2_x-ray.h5')
class_names = ['NORMAL','PNEUMONIA']
users = {}
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'DAT'}
MAX_CONTENT_LENGTH = 30 * 1024 * 1024  
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def import_and_predict(image_path, model):
    image = Image.open(image_path).convert('RGB')
    image = ImageOps.fit(image, (224, 224), Image.LANCZOS)
    img = np.asarray(image) / 255.0
    img = img[..., :3] if img.shape[-1] == 4 else img
    img_reshape = np.expand_dims(img, axis=0)
    predictions = model.predict(img_reshape)
    predicted_class_idx = np.argmax(predictions)
    return class_names[predicted_class_idx], predictions[0][predicted_class_idx]

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users:
            flash('Username already exists.', 'error')
            return redirect(url_for('signup'))
        users[username] = password
        flash('Signup successful!', 'success')
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username] == password:
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('predict'))
        flash('Invalid username or password.', 'error')
        return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/index')
def index():
    if 'username' in session:
        return render_template('index.html')
    flash('You need to log in first', 'error')
    return redirect(url_for('login'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            try:
                file_path = os.path.join('static/uploads', file.filename)
                file.save(file_path)
                predicted_class, accuracy = import_and_predict(file_path, model)
                return render_template('result.html', disease=predicted_class, accuracy=round(accuracy * 100, 2), real_image_path=f'/static/uploads/{file.filename}')
            except Exception as e:
                flash(f'Error: {str(e)}', 'error')
                return redirect(url_for('index'))
        flash('Invalid file format or file is too large.', 'error')
        return redirect(url_for('index'))
    return render_template('index.html')

@app.route('/performance')
def performance():
    labels = ['NORMAL','PNEUMONIA']  
    values = [1341, 3875]
    return render_template('performance.html', labels=labels, values=values)

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(port=5000, debug=True)
