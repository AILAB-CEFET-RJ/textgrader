# Textgrader
 
## Como rodar

Você pode rodar o projeto da seguinte forma:

Após clonar o repositório, instale os requirements (recomentadamos a criaçao de um ambiente virtual 
especifico no conda)

```
pip install -r requirements.txt
```

Após isso, para rodar o pipeline

```
kedro run --pipeline textgrader
```

## Para executar jupyter notebook

para rodar o jupyter notebook com acesso a recursos adicionais do kedro
```
kedro jupyter notebook 
```

para rodar jupyter notebook comum 
```
jupyter notebook 
```
 

## Conjuntos de texto

O conjunto de textos trabalhado contém cerca de 170 temas, não há uniformidade nos nomes dos conceitos avaliados e na escala numérica dos conceitos, por isso agrupamos os temas em três grupos diferentes que possuirão conceitos diferentes e escalas numéricas diferentes

### Primeiro conjunto de texto – temas 1 até 85 (cada conceito de  0 a200)
• 'Domínio da modalidade escrita formal',                                                                                                  
• 'Compreender a proposta e aplicar conceitos das várias áreas de conhecimento para desenvolver o texto dissertativo-argumentativo em prosa',   
• 'Selecionar, relacionar, organizar e interpretar informações em defesa de um ponto de vista',                                                 
• 'Conhecimento dos mecanismos linguísticos necessários para a construção da argumentação',                                                     
• 'Proposta de intervenção com respeito aos direitos humanos'           

### Segundo conjunto de texto -  temas 86 até 137 (cada conceito de 0 a 10)
•	'Conteúdo',                                                                                                                               
•	'Estrutura do texto',                                                                                                               
•	'Estrutura de ideias',                                                                                                                 
•	'Vocabulário',                                                                                                                       
•	'Gramática e ortografia' 

### Terceiro conjunto de texto - temas 137 – 170 (cada conceito de 0 a 10)
•	'Adequação ao Gênero Textual',                                                                                                               
•	'Adequação à modalidade padrão da língua',                                                                                                    
•	'Coesão e Coerência'
•	'Adequação ao Tema',                                                                                                                        
•	'Adequação e Leitura Crítica da Coletânea',                                                                                                


## Tecnologias utilizadas

Até o presente momento utilizamos:

*	Kedro – Framework de engenharia de software voltado a dados desenvolvido pela quantum black (braço de dados da consultoria mckinsey company) 
*	Pandas – biblioteca de manuseamento de dados
*	NLTK – para geração das features básicas
*	Sklearn para extração de features com tf-idf e computo dos resultados usando cohen_kappa
*	Xgboost modelo usado para treino 

O kedro em minha opinião me ajudou bastante a manter a organização do pipeline e ter um 
‘runner’ de forma fácil. Ele possui alguns conceitos centrais como nó, catalogo e pipeline, que para ilustrar melhor recomendo a consulta ao tutorial presente no seguinte link 
https://docs.kedro.org/en/stable/introduction/index.html


 

## Lógica do projeto 

consolidamos os diferentes JSONS com texto em 3 tabelas com os diferentes targets, 
após isso extraimos features básicas (quantidade de palavras, palavras únicas e sentenças),
na sequência, extraimos 1000 features usando TF-IDF, considerando um vocabulário formado pelas 3 tabelas.
Finalmente executamos dois experimentos

* treinar um modelo por tema e prever nesse tema
* treinar um modelo unico para todos os temas e prever em todos os temas

* os resultados já são gerados e podem ser consultados no caminho indicado pela entrada 
final_scores_experiment do arquivo conf/base/catalog.yml


Um esquema do pipeline será disponibilizado na proxima seção e, além disso, 
cada nó e cada função do código tem uma docstring com o que realiza e quais são seus argumentos,
além disso, grande parte dos trechos de código dentro das funções estão comentados

 
## visualizaçao do pipeline
Uma visualização do pipeline pode ser vista na Issue:
 
 https://github.com/RamonBoucas/textgrader_PT/issues/1

Aqui anexei uma imagem do pipeline, caso haja dificuldade em vê-la, essa imagem pode ser gerada 
utilizando o comando kedro-viz (após termos instalado as dependencias listadas em requirements.txt)