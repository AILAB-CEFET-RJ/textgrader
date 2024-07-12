import os
import json
from collections import defaultdict
import shutil


def agrupar_por_conjunto(diretorio):

    conjuntos = {}

    # Percorre todos os arquivos na pasta
    for nome_arquivo in os.listdir(diretorio):
        if nome_arquivo.endswith('.json'):
            caminho_arquivo = os.path.join(diretorio, nome_arquivo)

            with open(caminho_arquivo, 'r', encoding='utf-8') as arquivo:
                try:
                    dados = json.load(arquivo)

                    for objeto in dados:
                        competencias = objeto.get('competencias')
                        comps_list = []
                        for c in competencias:
                            comps_list.append(c['competencia'])

                        comps = "".join(comps_list)

                        if comps not in conjuntos.keys():
                            conjuntos[comps] = [caminho_arquivo]
                        else:
                            conjuntos[comps].append(caminho_arquivo)


                except json.JSONDecodeError as e:
                    print(f"Erro ao ler o arquivo {nome_arquivo}: {e}")

    return conjuntos


def mover_arquivo_para_pasta(caminho_arquivo, conj):
    # Cria a pasta para a nota se não existir
    pasta_nota = os.path.join(f'conjunto_{conj}')
    os.makedirs(pasta_nota, exist_ok=True)

    # Move o arquivo para a pasta correspondente à nota
    destino = os.path.join(pasta_nota, os.path.basename(caminho_arquivo))
    shutil.copy2(caminho_arquivo, destino)
    print(f"Arquivo {caminho_arquivo} movido para {destino}")


# Diretório onde estão os arquivos JSON
diretorio_dados = '.'

# Agrupa os objetos JSON pelo valor de 'nota'
agrupado_por_conj = agrupar_por_conjunto(diretorio_dados)

conjunto = 1
for c in agrupado_por_conj:
    for f in agrupado_por_conj[c]:
        mover_arquivo_para_pasta(f, conjunto)
    conjunto += 1
