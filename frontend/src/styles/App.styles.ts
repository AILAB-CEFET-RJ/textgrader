import styled from 'styled-components'
import { Footer } from 'antd/lib/layout/layout'

const MyFooter = styled(Footer)`
  text-align: center;
  color: #ffffff;
  gap: 6px;
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: #001529;
`

const NameLink = styled.a`
  text-decoration: none;
  color: #ffffff;
  font-weight: bold;
`

export const S = {
  MyFooter,
  NameLink
}
