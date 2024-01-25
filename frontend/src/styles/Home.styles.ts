import styled from 'styled-components'

const ContentWrapper = styled.div`
  display: flex;
  height: 84.7vh;
`

const TextWrapper = styled.div`
  display: flex;
  width: 51%;
  padding: 8%;
  justify-content: center;
  flex-direction: column;
  align-items: center;
`

const ImageContainer = styled.div`
  position: relative;
  object-fit: cover;
  width: 50%;
  max-width: 100%;
`

const Title = styled.h1`
  font-family: 'Castoro', serif;
  font-size: 3rem;
  margin-top: 20px;
`

const Description = styled.h4`
  font-size: 15px;
  font-weight: normal;
  line-height: 24px;
  text-align: justify;
`

const Divider = styled.div`
  border-top: solid 1px #AAAAAA;
  width: 30%;
  margin-bottom: 30px;
`

export const S = {
  ImageContainer,
  ContentWrapper,
  TextWrapper,
  Title,
  Description,
  Divider
}