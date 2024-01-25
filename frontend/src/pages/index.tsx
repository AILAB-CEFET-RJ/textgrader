import { S } from '@/styles/Home.styles'
import Image from 'next/image'
import BookPagesImg from '../../public/bookPages.jpg'

const Home = () => {
  return (
    <S.ContentWrapper>
      <S.ImageContainer className='ImageContainer'>
        <Image src={BookPagesImg} alt='Páginas de livros' fill style={{ objectFit: 'cover' }} />
      </S.ImageContainer>
      <S.TextWrapper>
          <S.Title>Text Grader</S.Title>
          <S.Divider />
          <S.Description>
            Na educação, há várias técnicas para avaliar os estudantes, como itens de múltipla escolha e itens
            discursivos. Os itens discursivos incluem Essays, semelhantes a redações, e Short Answer, que são respostas
            curtas. Essays podem ser sobre qualquer tema, mas avaliadores levam em conta aspectos de comunicação e
            raciocínio. Composition Essays são semelhantes a redações brasileiras, enquanto Examination Essays exigem
            que os alunos mostrem conhecimento acadêmico e o sintetizem em um curto espaço de tempo. Short Answers
            avaliam o conhecimento do aluno em tópicos específicos. Exames de larga escala, como o ENEM no Brasil e o
            NAEP nos EUA, usam itens discursivos, incluindo redações e respostas discursivas. A avaliação automática de
            itens discursivos é feita por programas de computador que atribuem conceitos a textos escritos em um
            contexto educacional.
          </S.Description>
      </S.TextWrapper>
    </S.ContentWrapper>
  )
}

export default Home
