import pandas as pd
import os
from scipy.stats import wilcoxon


def load_data(directory):
    data = {}
    for filename in os.listdir(directory):
        if filename.endswith(".csv") and filename.startswith("distances"):
            # Extraer dataset_id y penalty del nombre del archivo
            parts = filename.split('_')
            dataset_id = parts[1]
            penalty = parts[2].split('.')[0]

            # Cargar el archivo CSV en un DataFrame
            filepath = os.path.join(directory, filename)
            df = pd.read_csv(filepath, index_col=0)

            # Guardar el DataFrame en el diccionario
            data[(dataset_id, penalty)] = df
    return data


def wilcoxon_test_between_models(results_clg, results_nf):
    results = []
    for key in results_clg.keys():
        if key in results_nf:
            df1 = results_clg[key]
            df2 = results_nf[key]

            # Asumimos que las columnas de ambos DataFrames tienen los mismos n_vertex
            for column in df1.columns:
                stat, p_value = wilcoxon(df1[column], df2[column], alternative='greater')
                results.append({
                    'dataset': key[0],
                    'penalty': key[1],
                    'n_vertex': column,
                    'statistic': stat,
                    'p_value': p_value
                })
    return pd.DataFrame(results)


def wilcoxon_test_between_n_vertex(data):
    wcx_analysis = []
    for key, df in data.items():
        columns = df.columns
        for i in range(len(columns) - 1):
            for j in range(i + 1, len(columns)):
                stat, p_value = wilcoxon(df[columns[i]], df[columns[j]], alternative='greater')
                wcx_analysis.append({
                    'dataset': key[0],
                    'penalty': key[1],
                    'n_vertex_1': columns[i],
                    'n_vertex_2': columns[j],
                    'statistic': stat,
                    'p_value': p_value
                })
    return pd.DataFrame(wcx_analysis)

if __name__ == "__main__":
    exp_dir = "../../results/exp_1/"
    results_clg = load_data(exp_dir + 'clg/')
    results_nf = load_data(exp_dir + 'nf/')

    results = wilcoxon_test_between_models(results_clg, results_nf)
    print(results)
    results.to_csv('results_between_models.csv', index=False)

    results = wilcoxon_test_between_n_vertex(results_clg)
    print(results)
    results.to_csv('results_between_n_vertex.csv', index=False)