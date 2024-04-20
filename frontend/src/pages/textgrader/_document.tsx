import StyledComponentsRegistry from '@/lib/registry'
import { Html, Head, Main, NextScript } from 'next/document'

export default function Document() {
  return (
    <Html lang="en">
      <Head>
      <link rel="preconnect" href="https://fonts.googleapis.com" />
      <link rel="preconnect" href="https://fonts.gstatic.com" />
      <link href="https://fonts.googleapis.com/css2?family=Castoro:ital@0;1&display=swap" rel="stylesheet" />
      <link href="https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,700&display=swap" rel="stylesheet" />
      </Head>
      <body>
        <StyledComponentsRegistry>
          <Main />
          <NextScript />
        </StyledComponentsRegistry>
      </body>
    </Html>
  )
}
