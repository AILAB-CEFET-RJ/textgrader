import axios from 'axios';

import { useEffect, useState } from 'react';

import { Button, Modal, Skeleton } from 'antd';

import TextArea from 'antd/lib/input/TextArea';

const Redacao = () => {
    const [isModalOpen, setIsModalOpen] = useState(false);

    const showModal = async () => {
        await getEssayGrade();
        setIsModalOpen(true);
    };

    const handleOk = () => {
        setIsModalOpen(false);
    };

    const handleCancel = () => {
        setIsModalOpen(false);
    };

    const [essay, setEssay] = useState('');
    const [essayGrade, setEssayGrade] = useState<number | null>(null);

    const handleChange = (event: any) => {
        setEssay(event.target.value);
    };

    const getEssayGrade = async () => {
        const response = await axios.post('http://0.0.0.0:8000/text_grade/', {
            essay: essay,
        });

        const data = response.data;

        setEssayGrade(data.grade);
    }

    const clearEssay = () => {
        setEssay('');
    }

    return (
        <div style={{ padding: '0 50px' }}>
            <h1 style={{ textAlign: 'center' }}>Redação</h1>

            <TextArea value={essay} onChange={handleChange} style={{ padding: 24, minHeight: 380, background: 'white' }} placeholder="Escreva sua redação aqui" />

            <Button
                onClick={clearEssay}
                style={{ marginTop: '16px' }}
                type="primary"
                danger
            >
                Apagar redação
            </Button>

            <Button
                onClick={showModal}
                style={{ marginTop: '16px', marginLeft: '16px' }}
                type="primary"
            >
                Enviar redação
            </Button>

            <Modal title="Nota da redação" open={isModalOpen} onOk={handleOk} onCancel={handleCancel} footer={null}>
                {essayGrade ? `A nota da redação é ${essayGrade}` : <Skeleton paragraph={{ rows: 0 }} />}
            </Modal>
        </div>
    );
};

export default Redacao;