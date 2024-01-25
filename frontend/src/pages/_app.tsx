import '@/styles/globals.css'
import 'antd/dist/reset.css'
import { Header, Content } from 'antd/lib/layout/layout'
import type { AppProps } from 'next/app'
import Image from 'next/image'
import Head from 'next/head'
import { Layout, Menu } from 'antd'
import { InfoCircleOutlined, FileTextOutlined, HomeOutlined } from '@ant-design/icons'
import type { MenuProps } from 'antd'
import Link from 'next/link'
import { S } from '@/styles/App.styles'
import { useState } from 'react'
import cefetBranco from '../../public/cefetBranco.png'

const items: MenuProps['items'] = [
  {
    label: <Link href='/textgrader'>Início</Link>,
    key: 'inicio',
    icon: <HomeOutlined />,
  },
  {
    label: <Link href='/textgrader/redacao'>Redação</Link>,
    key: 'redacao',
    icon: <FileTextOutlined />,
  },
  {
    label: <Link href='/textgrader/sobre'>Sobre</Link>,
    key: 'sobre',
    icon: <InfoCircleOutlined />,
  },
]

export default function App({ Component, pageProps }: AppProps) {
  const [current, setCurrent] = useState('inicio')
  const github = 'https://github.com/'

  const onClick: MenuProps['onClick'] = e => {
    setCurrent(e.key)
  }

  return (
    <Layout style={{ minHeight: '100vh', display: 'flex', justifyContent: 'center' }}>
      <Head>
        <title>CEFET | Text Grader</title>
      </Head>
      <Header style={{ position: 'sticky', top: 0, zIndex: 1, padding: 0 }}>
        <Menu theme='dark' onClick={onClick} selectedKeys={[current]} mode='horizontal' items={items} />
      </Header>
      <Content style={{ alignSelf: 'center' }}>
        <Component {...pageProps} />
      </Content>
      <S.MyFooter>
        ©2023 Created by
        <S.NameLink href={`${github}cassiofb-dev`}>Cassio</S.NameLink>
        <S.NameLink href={`${github}juliemoura`}>Julie</S.NameLink>
        <S.NameLink href={`${github}Gustavo-Pettine`}>Pettine</S.NameLink>
      </S.MyFooter>
    </Layout>
  )
}
