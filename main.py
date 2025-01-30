!pip install pulp pandas
import pandas as pd
import pulp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression


def run_dea_analysis(input_file, output_file):
    # 1. Membaca data dari file txt
    #df = pd.read_csv(input_file, sep='\t')
    df = pd.read_csv(input_file)
    # 2. Preprocessing data
    df['Kecepatan Positif'] = 1/df['Output - Kecepatan Eksekusi (detik/request)']

    # 3. Definisi variabel input-output
    input_columns = [
        'Waktu (hours)',
        'Tenaga Kerja (man-hours)',
        'Pelatihan Model (hours)',
        'Data (GB)',
        'Infrastruktur IT (servers)',
        'Komputasi (vCPU/core)',
        'Memori Server (GB)'
    ]

    output_columns = [
        'Output - Ketahanan Model (1-10)',
        'Output - Kinerja Model (%)',
        'Kecepatan Positif'
    ]

    inputs = df[input_columns].values
    outputs = df[output_columns].values

    # 4. DEA Model
    efficiency_scores = []

    for dmu in range(len(df)):
        # Setup model optimisasi
        prob = pulp.LpProblem(f"DEA_DMU_{dmu}", pulp.LpMinimize)
        theta = pulp.LpVariable('theta', lowBound=0)
        lambdas = pulp.LpVariable.dicts('lambda', range(len(df)), lowBound=0)

        # Fungsi tujuan
        prob += theta

        # Kendala input
        for i in range(inputs.shape[1]):
            prob += pulp.lpSum([lambdas[j] * inputs[j][i] for j in range(len(df))]) <= theta * inputs[dmu][i]

        # Kendala output
        for r in range(outputs.shape[1]):
            prob += pulp.lpSum([lambdas[j] * outputs[j][r] for j in range(len(df))]) >= outputs[dmu][r]

        # Solusi
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        efficiency_scores.append(round(pulp.value(theta), 3))

    # 5. Menyimpan hasil
    df['Efisiensi'] = efficiency_scores
    result_df = df[['DMU', 'Efisiensi']].sort_values(by='Efisiensi', ascending=False)
    result_df.to_csv(output_file, sep='\t', index=False)
    print("Hasil Analisa DEA-CRS (Constant  Returns to Scale)")
    print(result_df)

def plot_benchmarking_peer_comparison(df, input_columns, output_columns):
    """
    Fungsi untuk memvisualisasikan benchmarking dan peer comparison.
    """
    # Hitung efisiensi menggunakan DEA-VRS
    inputs = df[input_columns].values
    outputs = df[output_columns].values
    
    # Dictionary untuk menyimpan peer comparison
    peer_comparison = {}
    
    for dmu in range(len(df)):
        prob = pulp.LpProblem(f"DEA_VRS_DMU_{dmu}", pulp.LpMinimize)
        theta = pulp.LpVariable('theta', lowBound=0)
        lambdas = pulp.LpVariable.dicts('lambda', range(len(df)), lowBound=0)
        
        # Fungsi tujuan
        prob += theta
        
        # Kendala input
        for i in range(inputs.shape[1]):
            prob += pulp.lpSum([lambdas[j] * inputs[j][i] for j in range(len(df))]) <= theta * inputs[dmu][i]
        
        # Kendala output
        for r in range(outputs.shape[1]):
            prob += pulp.lpSum([lambdas[j] * outputs[j][r] for j in range(len(df))]) >= outputs[dmu][r]
        
        # Kendala VRS
        prob += pulp.lpSum([lambdas[j] for j in range(len(df))]) == 1
        
        # Solusi optimasi
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        
        # Simpan nilai lambda (peer comparison)
        peer_comparison[df.iloc[dmu]['DMU']] = {
            df.iloc[j]['DMU']: round(pulp.value(lambdas[j]), 3) for j in range(len(df)) if pulp.value(lambdas[j]) > 0
        }
    
    # Plot benchmarking dan peer comparison
    plt.figure(figsize=(12, 8))
    for idx, (dmu, peers) in enumerate(peer_comparison.items()):
        peers_df = pd.DataFrame(list(peers.items()), columns=['Peer DMU', 'Lambda'])
        sns.barplot(data=peers_df, x='Peer DMU', y='Lambda', color='skyblue')
        plt.title(f'Benchmarking & Peer Comparison untuk {dmu}')
        plt.xlabel('Peer DMU')
        plt.ylabel('Kontribusi Lambda')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()


def run_dea_vrs_analysis(input_file, output_file):
    """
    Fungsi untuk menjalankan analisis DEA-VRS (Variable Returns to Scale).
    """
    # 1. Membaca data dari file input
    df = pd.read_csv(input_file)
    
    # 2. Preprocessing data
    df['Kecepatan Positif'] = 1 / df['Output - Kecepatan Eksekusi (detik/request)']
    
    # 3. Definisi variabel input-output
    input_columns = [
        'Waktu (hours)',
        'Tenaga Kerja (man-hours)',
        'Pelatihan Model (hours)',
        'Data (GB)',
        'Infrastruktur IT (servers)',
        'Komputasi (vCPU/core)',
        'Memori Server (GB)'
    ]
    output_columns = [
        'Output - Ketahanan Model (1-10)',
        'Output - Kinerja Model (%)',
        'Kecepatan Positif'
    ]
    
    inputs = df[input_columns].values
    outputs = df[output_columns].values
    
    # 4. DEA-VRS Model
    efficiency_scores = []
    for dmu in range(len(df)):
        prob = pulp.LpProblem(f"DEA_VRS_DMU_{dmu}", pulp.LpMinimize)
        theta = pulp.LpVariable('theta', lowBound=0)
        lambdas = pulp.LpVariable.dicts('lambda', range(len(df)), lowBound=0)
        prob += theta
        for i in range(inputs.shape[1]):
            prob += pulp.lpSum([lambdas[j] * inputs[j][i] for j in range(len(df))]) <= theta * inputs[dmu][i]
        for r in range(outputs.shape[1]):
            prob += pulp.lpSum([lambdas[j] * outputs[j][r] for j in range(len(df))]) >= outputs[dmu][r]
        prob += pulp.lpSum([lambdas[j] for j in range(len(df))]) == 1
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        efficiency_scores.append(round(pulp.value(theta), 3))
    
    # 5. Menyimpan hasil
    df['Efisiensi'] = efficiency_scores
    result_df = df[['DMU', 'Efisiensi']].sort_values(by='Efisiensi', ascending=False)
    result_df.to_csv(output_file, sep='\t', index=False)
    print("Hasil Analisa DEA-VRS (Variable Returns to Scale)")
    print(result_df)
    return df


def plot_correlation_heatmap(df):
    """
    Fungsi untuk memvisualisasikan korelasi antara variabel input/output dan efisiensi.
    """
    # Hitung korelasi
    correlation_matrix = df.corr(numeric_only=True)
    
    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Heatmap Korelasi Antar Variabel')
    plt.show()


def plot_regression_analysis(df):
    """
    Fungsi untuk memvisualisasikan pengaruh variabel terhadap efisiensi menggunakan regresi linear.
    """
    # Variabel independen (input/output) dan dependen (efisiensi)
    X = df.drop(columns=['DMU', 'Efisiensi'])
    y = df['Efisiensi']
    
    # Regresi linear
    model = LinearRegression()
    model.fit(X, y)
    
    # Koefisien regresi
    coefficients = pd.DataFrame({
        'Variabel': X.columns,
        'Koefisien': model.coef_
    }).sort_values(by='Koefisien', ascending=False)
    
    # Plot koefisien regresi
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Koefisien', y='Variabel', data=coefficients, palette='viridis')
    plt.title('Pengaruh Variabel Terhadap Efisiensi (Regresi Linear)')
    plt.xlabel('Koefisien Regresi')
    plt.ylabel('Variabel')
    plt.show()


def plot_sensitivity_analysis(df):
    """
    Fungsi untuk memvisualisasikan sensitivitas variabel terhadap efisiensi.
    """
    # Pilih variabel yang paling sensitif
    sensitivity_vars = ['Output - Kinerja Model (%)', 'Waktu (hours)', 'Pelatihan Model (hours)']
    sensitivity_df = df[sensitivity_vars + ['Efisiensi']]
    
    # Plot scatter plot untuk setiap variabel sensitif
    fig, axes = plt.subplots(1, len(sensitivity_vars), figsize=(18, 5))
    for i, var in enumerate(sensitivity_vars):
        sns.scatterplot(data=sensitivity_df, x=var, y='Efisiensi', ax=axes[i])
        axes[i].set_title(f'Sensitivitas {var}')
        axes[i].set_xlabel(var)
        axes[i].set_ylabel('Efisiensi')
    plt.tight_layout()
    plt.show()


def plot_slack_analysis(df):
    """
    Fungsi untuk memvisualisasikan slack (area yang perlu diperbaiki).
    """
    # Identifikasi slack (misalnya, input yang bisa dikurangi)
    slack_vars = ['Waktu (hours)', 'Tenaga Kerja (man-hours)', 'Pelatihan Model (hours)']
    slack_df = df[slack_vars + ['Efisiensi']]
    
    # Plot boxplot untuk slack
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=slack_df, orient='h', palette='Set2')
    plt.title('Analisis Slack: Area yang Perlu Diperbaiki')
    plt.xlabel('Nilai Input')
    plt.ylabel('Variabel Input')
    plt.show()


# Main execution
if __name__ == "__main__":
    # Jalankan analisis DEA-VRS
    df = run_dea_vrs_analysis('input.csv', 'output-vrs.txt')
    run_dea_analysis('input.csv', 'output-crs.txt')    
    # Plot heatmap korelasi
    plot_correlation_heatmap(df)
    
    # Plot regresi linear
    plot_regression_analysis(df)
    
    # Plot sensitivitas
    plot_sensitivity_analysis(df)
    
    # Plot slack
    plot_slack_analysis(df)



