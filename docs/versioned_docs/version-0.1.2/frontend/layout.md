---
sidebar_position: 2
---

# Layout

Let's check Textgrader **layout** in 5 minutes.

## Website Layout

```txt title="Layout Structure Tree"
- Layout
  - Menu Header
    - Navlink Menu Item: "Início"
    - Navlink Menu Item: "Redação"
    - Navlink Menu Item: "Sobre"
  - Content
  - Footer
    - Footer text
    - Footer links
```

The website layout is defined on the code bellow, in the next sub sections some sub set of this code will be explained.

```tsx title="frontend/src/pages/_app.tsx"
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
```

## Header Layout

The highlighted code bellow defines the website header.

```tsx title="frontend/src/pages/_app.tsx" {4-26}
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
```

## Content Layout

All website pages are wrapped into the ``Content`` element highlighted in the code bellow:

```tsx title="frontend/src/pages/_app.tsx"
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
      // highlight-next-line
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
```

## Footer Layout

The footer layout is highlighted in the code bellow:

```tsx title="frontend/src/pages/_app.tsx" {28-33}
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
```