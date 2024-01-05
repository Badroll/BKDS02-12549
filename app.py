# DEPLOY MENGGUNAKAN FLASK
# Hasil dari prediksi berupa model (termasuk informasi scaler) akan digunakan di sini


# menggunakan modul utama Flask sebagia web server
# web host menggunakan VPS
from flask import Flask, request, send_file, jsonify, render_template
from sklearn.model_selection import train_test_split
from os import path
import os
from datetime import datetime
import pickle
import pandas as pd
import hashlib
import json

# mendefinisikan app sebagai flask
app = Flask(__name__, template_folder="view")
acc = json.load(open("accuracy.json"))

# fungsi untuk return JSON (API)
def composeReply(status, message, payload = None):
    reply = {}
    reply["SENDER"] = "MSTH AI"
    reply["STATUS"] = status
    reply["MESSAGE"] = message
    reply["PAYLOAD"] = payload
    return jsonify(reply)

# menyimpan file
ALLOWED_EXTENSION = set(["csv", "xslx"])
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSION

def saveFile(file):
    try:
        #filename = str(datetime.now()).replace(":", "-") + (file.filename)
        filename = hashlib.md5(str(datetime.now()).encode('utf-8')).hexdigest() + "." + str(file.filename.rsplit(".", 1)[1].lower())
        basedir = path.abspath(path.dirname(__file__))
        file.save(path.join(basedir, "uploads", filename))
        return filename
    except TypeError as error : return [False, "Save file failed [" + error]

# menuju halaman index untuk memprediksi
@app.route("/", methods=['GET'])
def index():
    r = {
        "accuracy" : acc["accuracy"]
    }
    return render_template("prediksi.html", **r)


# memproses prediksi dan mengembalikannya
@app.route("/predict", methods=['POST'])
def predict():
    if request.form.get("type") == "single":
        # jika tipe prediksi tunggal
        age = request.form.get("age")
        sex = request.form.get("sex")
        cp = request.form.get("cp")
        trestbps = request.form.get("trestbps")
        chol = request.form.get("chol")
        fbs = request.form.get("fbs")
        restecg = request.form.get("restecg")
        thalach = request.form.get("thalach")
        exang = request.form.get("exang")
        oldpeak = request.form.get("oldpeak")
        
        # Data input
        new_data_dict = {
            'age': [age],
            'sex': [sex],
            'cp': [cp],
            'trestbps': [trestbps],
            'chol': [chol],
            'fbs': [fbs],
            'restecg': [restecg],
            'thalach': [thalach],
            'exang': [exang],
            'oldpeak': [oldpeak],
        }
        r = inference(new_data_dict)
        print(r)

    else:
        # jika tipe prediksi multi (banyak)
        file = request.files["file"]
        filename = saveFile(file)
        print(filename)

        file_path = f'uploads/{filename}'
        if ".xlsx" in filename:
            # Membaca file Excel
            df = pd.read_excel(file_path)
        elif ".csv" in filename:
            # Mengonversi DataFrame ke CSV
            df = pd.read_csv(file_path)

        # merender hasil dalam bentuk tabel
        html = """<style>
                    .styled-table {
                        width: 100%;
                        border-collapse: collapse;
                        margin: 25px 0;
                        font-size: 18px;
                        text-align: left;
                    }

                    .styled-table th,
                    .styled-table td {
                        padding: 12px 15px;
                    }

                    .styled-table th {
                        background-color: #f2f2f2;
                        font-weight: bold;
                    }

                    .styled-table tr:nth-child(even) {
                        background-color: #f5f5f5;
                    }

                    .styled-table tr:hover {
                        background-color: #e0e0e0;
                    }
                </style>
                <table class='styled-table'>
                    <thead>
                    <tr>
                        <th>age</th>
                        <th>sex</th>
                        <th>cp</th>
                        <th>trestbps</th>
                        <th>chol</th>
                        <th>fbs</th>
                        <th>restecg</th>
                        <th>thalach</th>
                        <th>exang</th>
                        <th>oldpeak</th>
                        <th>result</th>
                    </tr>
                    </thead>
                    <tbody>
                """

        # iterasi pada setiap baris DataFrame
        for index, row in df.iterrows():
            new_data_dict = {
                'age': [row['age']],
                'sex': [row['sex']],
                'cp': [row['cp']],
                'trestbps': [row['trestbps']],
                'chol': [row['chol']],
                'fbs': [row['fbs']],
                'restecg': [row['restecg']],
                'thalach': [row['thalach']],
                'exang': [row['exang']],
                'oldpeak': [row['oldpeak']],
            }
            test = inference(new_data_dict)
            html += "<tr>\n"
            html += f"""
                <td>{row['age']}</td>
                <td>{row['sex']}</td>
                <td>{row['cp']}</td>
                <td>{row['trestbps']}</td>
                <td>{row['chol']}</td>
                <td>{row['fbs']}</td>
                <td>{row['restecg']}</td>
                <td>{row['thalach']}</td>
                <td>{row['exang']}</td>
                <td>{row['oldpeak']}</td>
                <td>{test['info']}</td>
                \n"""
            html += "</tr>\n"
        html += """</tbody>
                </table>
            """
        
        r = {
            "results" : html
        }
        os.remove(f"uploads/{filename}")
    r["accuracy"] = acc["accuracy"]
    #return composeReply("SUCCESS", "Prediksi", r)
    return render_template("prediksi.html", **r)


# fungsi utama untuk memprediksi
def inference(new_data_dict):
    print(new_data_dict)

    # Memuat model dan scaler yang sudah disimpan
    loaded_model = pickle.load(open("model.sav", 'rb'))
    with open("scaler.pkl", 'rb') as file:
        scaler = pickle.load(file)

    # membuat DataFrame dari data input
    df = pd.DataFrame(new_data_dict)

    # mengonversi DataFrame menjadi array NumPy
    numpy_array = df.values.flatten().astype(float)

    # memastikan array memiliki dimensi yang sesuai dengan model (10 fitur)
    if numpy_array.shape[0] != 10:
        print("Error: Jumlah fitur tidak sesuai dengan model.")
    else:
        # reshape array menjadi bentuk yang sesuai dengan model (10 fitur)
        x = numpy_array.reshape(1, -1)

        # normalisasi menggunakan scaler
        x_normal = scaler.transform(x)

        # memprediksi dengan model yang sudah diload
        prediction = loaded_model.predict(x_normal)
        prediction = prediction.tolist()[0]
        ref = {
            0.0 : "Healthy",
            1.0 : "Heart disease level 1",
            2.0 : "Heart disease level 2",
            3.0 : "Heart disease level 3",
            4.0 : "Heart disease level 4",
        }
        print("Prediction:", prediction)

        # menampilkan hasil probabilitas
        prediction_proba = loaded_model.predict_proba(x_normal)
        print("Class Probabilities:", prediction_proba)

    r = {
        "class" : prediction,
        "info" : ref[prediction]
    }
    return r


if __name__ == '__main__':
    app.run(host = "0.0.0.0", port = 5005, debug = True)