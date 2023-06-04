import { useEffect, useState } from "react";

const Home = () => {


    return (
        <div style={{ display: "flex", justifyContent: "center", flexDirection: "column", alignItems: "center", height: "70vh", padding: "20px 50px", textAlign: "center", gap: "10px" }}>
            <h1>Text Grader</h1>
            <h4 style={{ fontSize: "15px", fontWeight: "normal", lineHeight: "24px" }}>
                Na educação, há várias técnicas para avaliar os estudantes, como itens de múltipla escolha e itens discursivos. Os itens discursivos incluem Essays, semelhantes a redações, e Short Answer, que são respostas curtas. Essays podem ser sobre qualquer tema, mas avaliadores levam em conta aspectos de comunicação e raciocínio. Composition Essays são semelhantes a redações brasileiras, enquanto Examination Essays exigem que os alunos mostrem conhecimento acadêmico e o sintetizem em um curto espaço de tempo. Short Answers avaliam o conhecimento do aluno em tópicos específicos. Exames de larga escala, como o ENEM no Brasil e o NAEP nos EUA, usam itens discursivos, incluindo redações e respostas discursivas. A avaliação automática de itens discursivos é feita por programas de computador que atribuem conceitos a textos escritos em um contexto educacional.
            </h4>
        </div>
    );
};

export default Home;