
TARGETS_1 = ['Domínio da modalidade escrita formal_nota',                                                                                                  
 'Selecionar, relacionar, organizar e interpretar informações em defesa de um ponto de vista_nota',                                                 
 'Compreender a proposta e aplicar conceitos das várias áreas de conhecimento para desenvolver o texto dissertativo-argumentativo em prosa_nota',   
'Conhecimento dos mecanismos linguísticos necessários para a construção da argumentação_nota',                                                     
'Proposta de intervenção com respeito aos direitos humanos_nota']           

TARGETS_1_RENAME = {'Domínio da modalidade escrita formal_nota':'c1',                                                                                                  
 'Compreender a proposta e aplicar conceitos das várias áreas de conhecimento para desenvolver o texto dissertativo-argumentativo em prosa_nota':'c2',   
 'Selecionar, relacionar, organizar e interpretar informações em defesa de um ponto de vista_nota':'c3',                                                 
'Conhecimento dos mecanismos linguísticos necessários para a construção da argumentação_nota':'c4',                                                     
'Proposta de intervenção com respeito aos direitos humanos_nota':'c5'}           



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

ID_VARS = ['index','tema','conjunto']

ALL_TARGETS = TARGETS_1 + TARGETS_2 + TARGETS_3 

EXCLUDE_COLS = ALL_TARGETS + ID_VARS
 
                                 