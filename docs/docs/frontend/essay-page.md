---
sidebar_position: 3
---

# Essay Page

Let's check Textgrader **most important page** in 5 minutes.

## Website Goal

The frontend goal is to send the user essay to the API and receive the grade from API, for this to be possible the essay page has a huge text area and a sent button. The essay page is described in the code bellow, some of it will be highlighted and explained in the next sections.

```tsx title="frontend/src/pages/redacao.tsx"
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
```

## Essay Page Features

### Essay Textarea

Textgrader essay textarea is a simple textarea that lets the user write the essays, as described in the codes bellow:

```tsx title="Essay Textarea"
<TextArea value={essay} onChange={handleChange} style={{ padding: 24, minHeight: 380, background: 'white' }} placeholder="Escreva sua redação aqui" />
```

### Erase Essay

Erase essay functionality is a button that quickly erase all user essay, bellow are the code that make it happen:

```tsx title="Erase Essay Button Layout"
<Button
  onClick={clearEssay}
  style={{ marginTop: '16px' }}
  type="primary"
  danger
>
  Apagar redação
</Button>
```

```tsx title="Erase Essay Button Function"
const clearEssay = () => {
  setEssay('');
}
```

### Send Essay

Last but not least is Textgrader main functionality that is send the user esssay to the backend API, this process happens in the code bellow:

```tsx title="Send Essay Button Layout"
<Button
  onClick={showModal}
  style={{ marginTop: '16px', marginLeft: '16px' }}
  type="primary"
>
  Enviar redação
</Button>
```

```tsx title="Essay Grade Modal"
const showModal = async () => {
  await getEssayGrade();
  setIsModalOpen(true);
};
```

```tsx title="Get Essay Grade Function"
const getEssayGrade = async () => {
  const response = await axios.post('http://0.0.0.0:8000/text_grade/', {
    essay: essay,
  });

  const data = response.data;

  setEssayGrade(data.grade);
}
```

```tsx title="Essay Grade Modal"
<Modal title="Nota da redação" open={isModalOpen} onOk={handleOk} onCancel={handleCancel} footer={null}>
  {essayGrade ? `A nota da redação é ${essayGrade}` : <Skeleton paragraph={{ rows: 0 }} />}
</Modal>
```
