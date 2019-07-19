import pandas as pd
import numpy as np

df = pd.read_csv('detrend.nino34.ascii2018.csv')

dados = pd.read_csv('results_corr_065_sem_modulo_mensal_sempolo_com_qtde_links.csv')
print(dados)
dados_media_classificado = []
dados_media = []

for i in df['ANOM ']:
    if i > 0.5:
        if i <= 1.5:
            dados_media_classificado.append(1)
        else:
            dados_media_classificado.append(2)
    else:
        dados_media_classificado.append(0)


dados['oni'] = df['ANOM ']
dados['flag'] = dados_media_classificado
dados.to_csv('results_corr_065_sem_modulo_mensal_sempolo_com_qtde_links_ajustado.csv')
print(dados)
        
    


