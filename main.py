from flask import Flask, render_template, request
import joblib
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
# Cargar el modelo entrenado
model = joblib.load('models/modelo_arbol.pkl')  # Cargar el modelo previamente guardado

# Crear una aplicación Flask
app = Flask(__name__)

# Definir la ruta principal del sitio web
@app.route('/')
def index():
    return render_template('index.html')  # Renderizar la plantilla 'index.html'


@app.route('/predict', methods=['POST'])
def predict():
    # Obtener los valores del formulario enviado
    fiebre = float(request.form['f'])
    tos = float(request.form['t'])
    dolor_garganta = float(request.form['dg'])
    congestion_nasal = float(request.form['cn'])
    dificultad_respiratoria = float(request.form['dr'])

    scaler = StandardScaler()
    
    # Valores futuros
    new_samples = np.array([[fiebre, tos, dolor_garganta, congestion_nasal, dificultad_respiratoria]])
    
    prediccion = model.predict(new_samples)
    # Iniciar la aplicación si este script es el punto de entrada
    return render_template('result.html', pred=prediccion[0]) 

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
