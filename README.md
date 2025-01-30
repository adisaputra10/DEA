# DEA
data envelopment Analysis


Fungsi untuk memvisualisasikan benchmarking dan peer comparison.

'''


 

    def plot_benchmarking_peer_comparison(df, input_columns, output_columns)
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


     # Definisi variabel input-output
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
    plot_benchmarking_peer_comparison(df, input_columns, output_columns)
'''
