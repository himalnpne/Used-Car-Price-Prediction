from flask import Flask, render_template, request, redirect, url_for, session, make_response, jsonify
import pandas as pd
import joblib
import sqlite3
import numpy as np
from sklearn.utils import resample
from datetime import datetime
import csv
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

app = Flask(__name__)
app.secret_key = 'your_secret_key'

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        self.tree_ = self._build_tree(X, y, depth=0)
    
    def _build_tree(self, X, y, depth):
        num_samples, num_features = X.shape
        if num_samples <= 1 or (self.max_depth is not None and depth >= self.max_depth):
            return np.mean(y)

        best_split = self._find_best_split(X, y)
        if best_split is None:
            return np.mean(y)
        
        left_indices = X[:, best_split['feature']] <= best_split['value']
        right_indices = X[:, best_split['feature']] > best_split['value']
        
        left_tree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_tree = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        
        return {'feature': best_split['feature'], 'value': best_split['value'], 'left': left_tree, 'right': right_tree}
    
    def _find_best_split(self, X, y):
        best_split = None
        best_mse = float('inf')
        num_features = X.shape[1]

        for feature in range(num_features):
            values = np.unique(X[:, feature])
            for value in values:
                left_indices = X[:, feature] <= value
                right_indices = X[:, feature] > value
                
                if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
                    continue
                
                left_y = y[left_indices]
                right_y = y[right_indices]
                
                mse = (np.var(left_y) * len(left_y) + np.var(right_y) * len(right_y)) / len(y)
                
                if mse < best_mse:
                    best_split = {'feature': feature, 'value': value}
                    best_mse = mse
        
        return best_split

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return np.array([self._predict(sample, self.tree_) for sample in X])
    
    def _predict(self, sample, tree):
        if not isinstance(tree, dict):
            return tree
        
        if sample[tree['feature']] <= tree['value']:
            return self._predict(sample, tree['left'])
        else:
            return self._predict(sample, tree['right'])

class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        for _ in range(self.n_estimators):
            X_resampled, y_resampled = resample(X, y)
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_resampled, y_resampled)
            self.trees.append(tree)
    
    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(tree_predictions, axis=0)

# Load the preprocessor and model
preprocessor = joblib.load('preprocessor.pkl')
model = joblib.load('car_price_model.pkl')

# Load the dataset for dropdown options
df = pd.read_csv('car_dataset.csv')
brands = df['brand'].unique()

def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

ADMIN_CREDENTIALS = {'username': 'admin', 'password': 'admin'}

@app.route('/')
def index():
    return render_template('index.html', brands=brands, user=session.get('username'))

@app.route('/predict_page')
def predict_page():
    return render_template('predict_page.html', brands=brands, user=session.get('username'))

@app.route('/get_models', methods=['POST'])
def get_models():
    selected_brand = request.form['brand']
    models = df[df['brand'] == selected_brand]['model'].unique()
    return jsonify({'models': list(models)})

@app.route('/terms_and_conditions')
def terms_and_conditions():
    return render_template('terms_and_conditions.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    try:
        data['mileage'] = float(data['mileage'])
        data['engine'] = float(data['engine'])
        data['max_power'] = float(data['max_power'])
        data['seats'] = int(data['seats'])
        data['vehicle_age'] = int(data['vehicle_age'])
        data['km_driven'] = int(data['km_driven'])
    except ValueError:
        return 'Invalid input data', 400
    
    selected_brand = data['brand']
    selected_model = data['model']
    car_name_series = df[(df['brand'] == selected_brand) & (df['model'] == selected_model)]['car_name']
    if not car_name_series.empty:
        data['car_name'] = car_name_series.values[0]
    else:
        return 'Car name not found in dataset', 404
    
    input_data = pd.DataFrame([data])
    
    # Ensure the input data has the same columns as the preprocessor expects
    expected_columns = preprocessor.feature_names_in_
    missing_columns = set(expected_columns) - set(input_data.columns)
    for col in missing_columns:
        input_data[col] = 0  # or some default value
    
    input_data_preprocessed = preprocessor.transform(input_data)
    
    # Debugging: Check the shape of the preprocessed data
    print(f"Preprocessed data shape: {input_data_preprocessed.shape}")
    
    try:
        prediction = model.predict(input_data_preprocessed)
    except ValueError as e:
        return f"Prediction error: {str(e)}", 500
    
    # Round off the prediction to the nearest whole number
    prediction = round(prediction[0])
    
    # Fetch the actual price from the dataset
    actual_price = df[(df['brand'] == selected_brand) & 
                      (df['model'] == selected_model) & 
                      (df['transmission_type'] == data['transmission_type']) & 
                      (df['fuel_type'] == data['fuel_type']) & 
                      (df['seller_type'] == data['seller_type']) & 
                      (df['km_driven'] == data['km_driven']) & 
                      (df['mileage'] == data['mileage']) & 
                      (df['engine'] == data['engine']) & 
                      (df['max_power'] == data['max_power']) & 
                      (df['seats'] == data['seats']) & 
                      (df['vehicle_age'] == data['vehicle_age'])]['selling_price'].values
    
    if actual_price.size > 0:
        actual_price = actual_price[0]
    else:
        actual_price = 'N/A'
    
    # Save predicted price to prediction_history table
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('''
        INSERT INTO prediction_history (user_id, brand, model, transmission_type, fuel_type, seller_type, km_driven, mileage, engine, max_power, seats, vehicle_age, predicted_price)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (session.get('user_id'), data['brand'], data['model'], data['transmission_type'], data['fuel_type'], data['seller_type'], data['km_driven'], data['mileage'], data['engine'], data['max_power'], data['seats'], data['vehicle_age'], prediction))
        
        # If the user is logged in, also save to history table
        if 'user_id' in session:
            cursor.execute('''
            INSERT INTO history (username, brand, model, transmission_type, fuel_type, seller_type, km_driven, mileage, engine, max_power, seats, vehicle_age, prediction)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (session.get('username'), data['brand'], data['model'], data['transmission_type'], data['fuel_type'], data['seller_type'], data['km_driven'], data['mileage'], data['engine'], data['max_power'], data['seats'], data['vehicle_age'], prediction))
        
        conn.commit()
        print("Prediction saved to database successfully")
    except Exception as e:
        print(f"Error saving prediction to database: {str(e)}")
        conn.rollback()
    finally:
        conn.close()
    
    # Fetch similar cars based on the predicted price
    similar_cars = fetch_similar_cars(data, prediction)
    
    # Convert models to a list
    models = df[df['brand'] == selected_brand]['model'].unique().tolist()
    
    return render_template('predict_page.html', prediction=prediction, brands=brands, models=models, user=session.get('username'), similar_cars=similar_cars)

def fetch_similar_cars(data, predicted_price):
    # Filter the dataset for similar cars based on predicted price
    similar_cars = df[(df['selling_price'] >= predicted_price * 0.9) & 
                      (df['selling_price'] <= predicted_price * 1.1) & 
                      (df['brand'] == data['brand']) & 
                      (df['model'] == data['model']) & 
                      (df['transmission_type'] == data['transmission_type']) & 
                      (df['fuel_type'] == data['fuel_type']) & 
                      (df['seller_type'] == data['seller_type']) & 
                      (df['seats'] == data['seats']) & 
                      (df['vehicle_age'] >= data['vehicle_age'] - 1) & 
                      (df['vehicle_age'] <= data['vehicle_age'] + 1)]
    
    # Limit the number of similar cars to display
    similar_cars = similar_cars.head(5)
    
    # Drop the 'car_name' column
    similar_cars = similar_cars.drop(columns=['car_name'])
    
    return similar_cars.to_dict(orient='records')

@app.route('/save_car_details', methods=['POST'])
def save_car_details():
    if 'username' not in session:
        return jsonify({'status': 'error', 'message': 'User not logged in'}), 401
    
    data = request.json
    
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('''
        INSERT INTO saved_cars (user_id, brand, model, transmission_type, fuel_type, seller_type, km_driven, mileage, engine, max_power, seats, vehicle_age, predicted_price)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (session.get('user_id'), data['brand'], data['model'], data['transmission_type'], data['fuel_type'], data['seller_type'], data['km_driven'], data['mileage'], data['engine'], data['max_power'], data['seats'], data['vehicle_age'], data['predicted_price']))
        
        conn.commit()
        print("Car details saved to database successfully")
        return jsonify({'status': 'success', 'message': 'Car details saved successfully'})
    except Exception as e:
        print(f"Error saving car details to database: {str(e)}")
        conn.rollback()
        return jsonify({'status': 'error', 'message': 'Failed to save car details'}), 500
    finally:
        conn.close()

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM history WHERE username = ?', (session['username'],))
    history = cursor.fetchall()
    cursor.execute('SELECT email, phone_number FROM users WHERE username = ?', (session['username'],))
    user_info = cursor.fetchone()
    cursor.execute('SELECT * FROM saved_cars WHERE user_id = ?', (session['user_id'],))
    saved_cars = cursor.fetchall()
    conn.close()
    
    if user_info is None:
        return "User information not found", 404
    
    user = {
        'username': session['username'],
        'email': user_info['email'],
        'phone_number': user_info['phone_number']
    }
    
    # Fetch actual prices for saved cars from the dataset
    saved_cars_with_prices = []
    for car in saved_cars:
        car_dict = dict(car)  # Convert sqlite3.Row to a regular dictionary
        actual_price = df[(df['brand'] == car_dict['brand']) & 
                          (df['model'] == car_dict['model']) & 
                          (df['transmission_type'] == car_dict['transmission_type']) & 
                          (df['fuel_type'] == car_dict['fuel_type']) & 
                          (df['seller_type'] == car_dict['seller_type']) & 
                          (df['km_driven'] == car_dict['km_driven']) & 
                          (df['mileage'] == car_dict['mileage']) & 
                          (df['engine'] == car_dict['engine']) & 
                          (df['max_power'] == car_dict['max_power']) & 
                          (df['seats'] == car_dict['seats']) & 
                          (df['vehicle_age'] == car_dict['vehicle_age'])]['selling_price'].values
        
        if actual_price.size > 0:
            car_dict['actual_price'] = actual_price[0]
        else:
            car_dict['actual_price'] = 'N/A'
        
        saved_cars_with_prices.append(car_dict)
    
    return render_template('dashboard.html', history=history, user=user, saved_cars=saved_cars_with_prices)

@app.route('/logout')
def logout():
    # Clear all session data
    session.clear()
    
    # Redirect to the index page
    resp = make_response(redirect(url_for('index')))
    resp.delete_cookie('username')
    return resp

@app.route('/admin', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == ADMIN_CREDENTIALS['username'] and password == ADMIN_CREDENTIALS['password']:
            session['admin_logged_in'] = True
            return redirect(url_for('admin_dashboard'))
        else:
            return "Invalid credentials", 401

    return render_template('admin_login.html')

@app.route('/admin/dashboard')
def admin_dashboard():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users')
    all_users = cursor.fetchall()
    conn.close()
    
    return render_template('admin_dashboard.html', all_users=all_users)

@app.route('/admin/edit_user/<username>', methods=['GET', 'POST'])
def edit_user(username):
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    
    conn = get_db_connection()
    cursor = conn.cursor()
    if request.method == 'POST':
        new_username = request.form['username']
        new_password = request.form['password']
        new_email = request.form['email']
        new_phone_number = request.form['phone_number']
        
        cursor.execute('''
        UPDATE users
        SET username = ?, password = ?, email = ?, phone_number = ?
        WHERE username = ?
        ''', (new_username, new_password, new_email, new_phone_number, username))
        conn.commit()
        conn.close()
        return redirect(url_for('admin_dashboard'))
    
    cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
    user = cursor.fetchone()
    conn.close()
    
    return render_template('edit_user.html', user=user)

@app.route('/delete_history/<int:history_id>', methods=['POST'])
def delete_history(history_id):
    if 'username' not in session:
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM history WHERE id = ? AND username = ?', (history_id, session['username']))
    conn.commit()
    conn.close()
    
    return redirect(url_for('dashboard'))

@app.route('/admin/delete_user/<username>', methods=['POST'])
def delete_user(username):
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM users WHERE username = ?', (username,))
    conn.commit()
    conn.close()
    
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/prediction_history', methods=['GET'])
def admin_prediction_history():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    
    filter_option = request.args.get('filter', default='all', type=str)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    if filter_option == 'logged_in':
        cursor.execute('''
            SELECT ph.* 
            FROM prediction_history ph
            JOIN users u ON ph.user_id = u.id
            WHERE u.username IS NOT NULL
            ORDER BY ph.timestamp DESC
        ''')
    elif filter_option == 'not_logged_in':
        cursor.execute('''
            SELECT ph.* 
            FROM prediction_history ph
            LEFT JOIN users u ON ph.user_id = u.id
            WHERE u.username IS NULL
            ORDER BY ph.timestamp DESC
        ''')
    else:
        cursor.execute('SELECT * FROM prediction_history ORDER BY timestamp DESC')
    
    prediction_history = cursor.fetchall()
    conn.close()
    
    return render_template('admin_prediction_history.html', prediction_history=prediction_history, filter_option=filter_option)

@app.route('/admin/export_prediction_history', methods=['GET'])
def export_prediction_history():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    
    filter_option = request.args.get('filter', default='all', type=str)
    format = request.args.get('format', default='csv', type=str)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    if filter_option == 'logged_in':
        cursor.execute('''
            SELECT ph.* 
            FROM prediction_history ph
            JOIN users u ON ph.user_id = u.id
            WHERE u.username IS NOT NULL
            ORDER BY ph.timestamp DESC
        ''')
    elif filter_option == 'not_logged_in':
        cursor.execute('''
            SELECT ph.* 
            FROM prediction_history ph
            LEFT JOIN users u ON ph.user_id = u.id
            WHERE u.username IS NULL
            ORDER BY ph.timestamp DESC
        ''')
    else:
        cursor.execute('SELECT * FROM prediction_history ORDER BY timestamp DESC')
    
    prediction_history = cursor.fetchall()
    conn.close()

    if format == 'csv':
        return export_csv(prediction_history)
    elif format == 'pdf':
        return export_pdf(prediction_history)
    else:
        return 'Invalid format', 400

@app.route('/delete_saved_car/<int:car_id>', methods=['POST'])
def delete_saved_car(car_id):
    if 'username' not in session:
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM saved_cars WHERE id = ? AND user_id = ?', (car_id, session['user_id']))
    conn.commit()
    conn.close()
    
    return redirect(url_for('dashboard'))

@app.route('/admin/export_users', methods=['GET'])
def export_users():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users')
    users = cursor.fetchall()
    conn.close()

    format = request.args.get('format', default='csv', type=str)

    if format == 'csv':
        return export_users_csv(users)
    elif format == 'pdf':
        return export_users_pdf(users)
    else:
        return 'Invalid format', 400

def export_users_csv(data):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Username', 'Email', 'Phone Number'])
    for row in data:
        writer.writerow([row['username'], row['email'], row['phone_number']])
    
    output.seek(0)
    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = "attachment; filename=users.csv"
    response.headers["Content-type"] = "text/csv"
    return response

def export_users_pdf(data):
    output = io.BytesIO()
    p = canvas.Canvas(output, pagesize=letter)
    width, height = letter

    text = p.beginText(40, height - 40)
    text.setFont("Helvetica-Bold", 12)
    text.textLine('User Information')
    text.setFont("Helvetica", 10)
    text.textLine('Username | Email | Phone Number')

    for row in data:
        text.textLine(f"{row['username']} | {row['email']} | {row['phone_number']}")

    p.drawText(text)
    p.showPage()
    p.save()

    output.seek(0)
    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = "attachment; filename=users.pdf"
    response.headers["Content-type"] = "application/pdf"
    return response

def export_csv(data):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['User ID', 'Brand', 'Model', 'Transmission Type', 'Fuel Type', 'Seller Type', 'KM Driven', 'Mileage', 'Engine', 'Max Power', 'Seats', 'Vehicle Age', 'Predicted Price', 'Timestamp'])
    for row in data:
        writer.writerow([row['user_id'], row['brand'], row['model'], row['transmission_type'], row['fuel_type'], row['seller_type'], row['km_driven'], row['mileage'], row['engine'], row['max_power'], row['seats'], row['vehicle_age'], row['predicted_price'], row['timestamp']])
    
    output.seek(0)
    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = "attachment; filename=prediction_history.csv"
    response.headers["Content-type"] = "text/csv"
    return response

def export_pdf(data):
    output = io.BytesIO()
    p = canvas.Canvas(output, pagesize=letter)
    width, height = letter

    text = p.beginText(40, height - 40)
    text.setFont("Helvetica-Bold", 12)
    text.textLine('Prediction History')
    text.setFont("Helvetica", 10)
    text.textLine('User ID | Brand | Model | Transmission Type | Fuel Type | Seller Type | KM Driven | Mileage | Engine | Max Power | Seats | Vehicle Age | Predicted Price | Timestamp')

    for row in data:
        text.textLine(f"{row['user_id']} | {row['brand']} | {row['model']} | {row['transmission_type']} | {row['fuel_type']} | {row['seller_type']} | {row['km_driven']} | {row['mileage']} | {row['engine']} | {row['max_power']} | {row['seats']} | {row['vehicle_age']} | {row['predicted_price']} | {row['timestamp']}")

    p.drawText(text)
    p.showPage()
    p.save()

    output.seek(0)
    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = "attachment; filename=prediction_history.pdf"
    response.headers["Content-type"] = "application/pdf"
    return response

@app.route('/user/export_prediction_history', methods=['GET'])
def user_export_prediction_history():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM prediction_history WHERE user_id = ?', (session['user_id'],))
    prediction_history = cursor.fetchall()
    conn.close()

    format = request.args.get('format', default='csv', type=str)

    if format == 'csv':
        return export_csv(prediction_history)
    elif format == 'pdf':
        return export_pdf(prediction_history)
    else:
        return 'Invalid format', 400

@app.route('/update_user_details', methods=['POST'])
def update_user_details():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    new_username = request.form['username']
    new_email = request.form['email']
    new_phone_number = request.form['phone_number']
    new_password = request.form['password']
    confirm_password = request.form['confirm_password']
    
    if new_password != confirm_password:
        return "Passwords do not match", 400
    
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('''
        UPDATE users
        SET username = ?, email = ?, phone_number = ?, password = ?
        WHERE username = ?
        ''', (new_username, new_email, new_phone_number, new_password, session['username']))
        conn.commit()
        session['username'] = new_username
        print("User details updated successfully")
    except Exception as e:
        print(f"Error updating user details: {str(e)}")
        conn.rollback()
    finally:
        conn.close()
    
    return redirect(url_for('dashboard'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password))
        user = cursor.fetchone()
        conn.close()
        
        if user:
            session['username'] = username
            session['user_id'] = user['id']
            resp = make_response(redirect(url_for('index')))
            resp.set_cookie('username', username)
            return resp
        else:
            return 'Invalid credentials', 401
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        phone_number = request.form['phone_number']
        
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
            INSERT INTO users (username, password, email, phone_number)
            VALUES (?, ?, ?, ?)
            ''', (username, password, email, phone_number))
            conn.commit()
            conn.close()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            conn.close()
            return 'Username already exists', 409
    
    return render_template('signup.html')

@app.route('/saved_cars')
def saved_cars():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM saved_cars WHERE user_id = ?', (session['user_id'],))
    saved_cars = cursor.fetchall()
    conn.close()
    
    # Fetch actual prices for saved cars from the dataset
    saved_cars_with_prices = []
    for car in saved_cars:
        car_dict = dict(car)  # Convert sqlite3.Row to a regular dictionary
        actual_price = df[(df['brand'] == car_dict['brand']) & 
                          (df['model'] == car_dict['model']) & 
                          (df['transmission_type'] == car_dict['transmission_type']) & 
                          (df['fuel_type'] == car_dict['fuel_type']) & 
                          (df['seller_type'] == car_dict['seller_type']) & 
                          (df['km_driven'] == car_dict['km_driven']) & 
                          (df['mileage'] == car_dict['mileage']) & 
                          (df['engine'] == car_dict['engine']) & 
                          (df['max_power'] == car_dict['max_power']) & 
                          (df['seats'] == car_dict['seats']) & 
                          (df['vehicle_age'] == car_dict['vehicle_age'])]['selling_price'].values
        
        if actual_price.size > 0:
            car_dict['actual_price'] = actual_price[0]
        else:
            car_dict['actual_price'] = 'N/A'
        
        saved_cars_with_prices.append(car_dict)
    
    return render_template('saved_cars.html', saved_cars=saved_cars_with_prices, user=session.get('username'))

if __name__ == '__main__':
    app.run(debug=True)