import streamlit as st
import numpy as np
import pandas as pd
import math
import joblib

tuning = pd.read_csv('hasiltuning/hasiltuning.csv')
best_index = tuning['mse'].idxmin()
best_hidden_layer_global = tuning['hidden_layer'][best_index]
best_hidden_neuron_global = tuning['hidden_neuron'][best_index]
best_aktivasi_global = tuning['aktivasi'][best_index]
best_lr_global = tuning['lr'][best_index]
best_epoch_global = tuning['epoch'][best_index]
best_mse_global = tuning['mse'][best_index]
best_bobot_global = tuning['bobot'][best_index]
best_bias_global = tuning['bias'][best_index]
best_bobot_output_global = tuning['bobot_output'][best_index]
best_bias_output_global = tuning['bias_output'][best_index]

best_bobot_global = eval(best_bobot_global)
best_bias_global = eval(best_bias_global)
best_bobot_output_global = eval(best_bobot_output_global)


def aktivasiX(activ, x): 
    # aktivasi relu
    if activ == 'relu':
        if x >= 0:
            return x
        else:
            return 0
    # aktivasi sigmoid
    elif activ == 'sigmoid':
        return 1 / (1 + math.exp(-x))
    # aktivasi softmax
    elif activ == 'softmax':
        return 1
    # aktivasi tanh
    else:
        return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

def turunanaktivasiX(activ, x): 
    # turunan aktivasi relu
    if activ == 'relu':
        if x > 0:
            return 1
        else:
            return 0
    # turunan aktivasi sigmoid
    elif activ == 'sigmoid':
        sig = 1 / (1 + math.exp(-x))
        return sig * (1 - sig)
    # turunan aktivasi softmax
    elif activ == 'softmax':
        return 0
    # turunan aktivasi tanh
    else:
        tan = (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))
        return 1 - (tan) ** 2

def predict(X_new, bobot, bias, bobot_output, bias_output, aktivasi):
    prediksi = []
    for i in range(len(X_new)):
        # FEEDFORWARD
        # Operasi pada Hidden Layer
        aktivasi_hidden = []
        for j in range(len(bobot)):  # untuk setiap hidden layer
            aktivasi_hidden_temp = []
            for k in range(len(bobot[j][0])):  # jumlah neuron di hidden layer
                sumXbobot = 0
                if j == 0:  # layer pertama
                    for l in range(4):  # jumlah neuron input
                        sumXbobot += bobot[j][l][k] * X_new[i][l]
                else:
                    for l in range(len(aktivasi_hidden[j - 1])):
                        sumXbobot += bobot[j][l][k] * aktivasi_hidden[j - 1][l]
                sumXbobotbias = bias[j][k] + sumXbobot
                aktivasi_hidden_temp.append(aktivasiX(aktivasi, sumXbobotbias))
            aktivasi_hidden.append(aktivasi_hidden_temp)

        # Operasi pada Output Layer
        sumZbobotoutput = 0
        for j in range(len(aktivasi_hidden[-1])):
            sumZbobotoutput += bobot_output[j] * aktivasi_hidden[-1][j]
        sumZbobotoutput_biasoutput = bias_output + sumZbobotoutput
        predik = aktivasiX(aktivasi, sumZbobotoutput_biasoutput)
        prediksi.append(predik)
    return prediksi


st.set_page_config(
    page_title="Aplikasi Prediksi Angka IPM",
    layout="wide"
)


st.title("Selamat datang di :red[Aplikasi Prediksi Angka Indeks Pembangunan Manusia]")
st.write("---")
st.markdown('##### Masukkan data berikut untuk melakukan prediksi IPM.')

AHH = st.number_input(label=r"$\textsf{\large Angka Harapan Hidup}$", value=0.0, min_value=0.0)
RLS = st.number_input(label=r"$\textsf{\large Rerata Lama Sekolah}$", value=0.0, min_value=0.0)
HLS = st.number_input(label=r"$\textsf{\large Harapan Lama Sekolah}$", value=0.0, min_value=0.0)
PK = st.number_input(label=r"$\textsf{\large Pengeluaran per Kapita}$", value=0.0, min_value=0.0)

inputs = np.array([[AHH, RLS, HLS, PK]])

ok = st.button("Prediksi")
if ok:
    scaler_fitur = joblib.load('model scaling/fitur_mmscaler_model.pkl')
    X_new_scaled = scaler_fitur.transform(inputs)
    hasil_prediksi = predict(X_new_scaled, best_bobot_global, best_bias_global, best_bobot_output_global, best_bias_output_global, best_aktivasi_global)
    scaler_target = joblib.load('model scaling/target_mmscaler_y.pkl')
    y_pred_original = scaler_target.inverse_transform([[hasil_prediksi[0]]])
    # pengkategorian IPM
    if y_pred_original[0][0] >= 80:
        kategori = 'Sangat Tinggi'
    elif 70 <= y_pred_original[0][0] < 80:
        kategori = 'Tinggi'
    elif 60 <= y_pred_original[0][0] < 70:
        kategori = 'Sedang'
    else:
        kategori = 'Rendah'
    st.write("## IPM yang diprediksi sebesar", y_pred_original[0][0])
    st.write("## Kategori IPM", kategori)

    # to run: streamlit run Homepage.py