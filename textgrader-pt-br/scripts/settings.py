TARGETS_1 = ['Domínio da modalidade escrita formal_nota',                                                                                                  
 'Selecionar, relacionar, organizar e interpretar informações em defesa de um ponto de vista_nota',                                                 
 'Compreender a proposta e aplicar conceitos das várias áreas de conhecimento para desenvolver o texto dissertativo-argumentativo em prosa_nota',   
'Conhecimento dos mecanismos linguísticos necessários para a construção da argumentação_nota',                                                     
'Proposta de intervenção com respeito aos direitos humanos_nota']           

TARGETS_3 = ['Conteúdo_nota',                                                                                                                               
'Estrutura do texto_nota',                                                                                                               
'Estrutura de ideias_nota',                                                                                                                 
'Vocabulário_nota',                                                                                                                       
'Gramática e ortografia_nota'] 

TARGETS_2 =  ['Adequação ao Tema_nota',                                                                                                                        
 'Adequação e Leitura Crítica da Coletânea_nota',                                                                                                
 'Adequação ao Gênero Textual_nota',                                                                                                               
 'Adequação à modalidade padrão da língua_nota',                                                                                                    
 'Coesão e Coerência_nota']

COLUMNS_REPLACE = {
 'Domínio da modalidade escrita formal':"c1",
 'Selecionar, relacionar, organizar e interpretar informações em defesa de um ponto de vista':"c2",
 'Compreender a proposta e aplicar conceitos das várias áreas de conhecimento para desenvolver o texto dissertativo-argumentativo em prosa':"c3",
'Conhecimento dos mecanismos linguísticos necessários para a construção da argumentação':"c4",
'Proposta de intervenção com respeito aos direitos humanos':"c5",
'Conteúdo':"c6",
'Estrutura do texto':"c7",
'Estrutura de ideias':"c8",
'Vocabulário':"c9",
'Gramática e ortografia':"c10",
'Adequação ao Tema':"c11",
 'Adequação e Leitura Crítica da Coletânea':"c12",
 'Adequação ao Gênero Textual':"c13",
 'Adequação à modalidade padrão da língua':"c14",
 'Coesão e Coerência':"c15"
}

TARGET_1_REPLACED = ["c1","c2","c3","c4","c5"]

ID_VARS = ['index','tema','conjunto']

ALL_TARGETS = TARGET_1_REPLACED

EXCLUDE_COLS = ALL_TARGETS + ID_VARS
 
                                 
TF_IDF_MAX_FEATURES = [32,64]

DATA_PATH = "../jsons/*.json"
OUTPUT_DF = "../data_one_label/"

DF_PATH = "../data/dataframe.csv"
