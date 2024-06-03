import axios from 'axios'
import { useState } from 'react'
import { Modal, Skeleton } from 'antd'
import { ClearOutlined, CheckOutlined } from '@ant-design/icons'
import TextArea from 'antd/lib/input/TextArea'
import { S } from '@/styles/Redacao.styles'

const Redacao = () => {
    const [isModalOpen, setIsModalOpen] = useState(false)
    const [essay, setEssay] = useState('')
    const [essayGrade, setEssayGrade] = useState<object>({ key: 'value' })

  const showModal = async () => {
    await getEssayGrade()
    setIsModalOpen(true)
  }

  const handleOk = () => {
    setIsModalOpen(false)
  }

  const handleCancel = () => {
    setIsModalOpen(false)
  }

  const handleChange = (event: any) => {
    setEssay(event.target.value)
  }

  const getEssayGrade = async () => {
    const response = await axios.post('http://localhost:5000/model', {
      essay: essay,
    })

    const data = response.data
    console.log(data.grades)

    setEssayGrade(data.grades)
  }

  const clearEssay = () => {
    setEssay('')
  }

  return (
    <S.Wrapper>
      <S.Title>🧾 Redação 🧾</S.Title>
      <TextArea
        value={essay}
        onChange={handleChange}
        style={{ padding: 24, minHeight: 380, background: 'white', width: '100%' }}
        placeholder='Escreva sua redação aqui'
      />

      <S.ButtonWrapper>
        <S.MyButton onClick={clearEssay} size='large' type='primary' danger icon={<ClearOutlined />}>
          Apagar texto
        </S.MyButton>

        <S.MyButton onClick={showModal} size='large' type='primary' icon={<CheckOutlined />}>
          Obter nota
        </S.MyButton>
      </S.ButtonWrapper>

      <Modal title='Nota da redação' open={isModalOpen} onOk={handleOk} onCancel={handleCancel} footer={null}>
        {essayGrade ? (
          Object.entries(essayGrade).map(([key, value], index) => (
            <p key={index}>{key}: {value}</p>
          ))
        ) : (
          <Skeleton paragraph={{ rows: 0 }} />
        )}
      </Modal>
    </S.Wrapper>
  )
}

export default Redacao
