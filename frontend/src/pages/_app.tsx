import '@/styles/globals.css';

import 'antd/dist/reset.css';

import { Header, Content, Footer } from 'antd/lib/layout/layout';

import type { AppProps } from 'next/app';

import { Layout, Menu } from 'antd';

import Link from 'next/link';

export default function App({ Component, pageProps }: AppProps) {
  return (
    <Layout style={{ minHeight: "100vh" }}>
      <Header style={{ position: 'sticky', top: 0, zIndex: 1, width: '100%' }}>
        <Menu
          theme="dark"
          mode="horizontal"
          defaultSelectedKeys={['2']}
        >
          <Menu.Item>
            <Link href="/">
              Início
            </Link>
          </Menu.Item>
          <Menu.Item>
            <Link href="/redacao">
              Redação
            </Link>
          </Menu.Item>
          <Menu.Item>
            <Link href="/sobre">
              Sobre
            </Link>
          </Menu.Item>
        </Menu>
      </Header>
      <Content style={{ padding: '20px 0' }} ><Component {...pageProps} /></Content>
      <Footer style={{ textAlign: 'center', gap: "6px", display: "flex", alignItems: "center", justifyContent: "center" }}>
        ©2023 Created by
        <a href="https://github.com/cassiofb-dev" style={{ textDecoration: "none", color: "#000", fontWeight: "bold" }}>Cassio</a>
        <a href="https://github.com/juliemoura" style={{ textDecoration: "none", color: "#000", fontWeight: "bold" }}>Julie</a>
        <a href="https://github.com/Gustavo-Pettine" style={{ textDecoration: "none", color: "#000", fontWeight: "bold" }}>Pettine</a>
      </Footer>
    </Layout>
  )
}
