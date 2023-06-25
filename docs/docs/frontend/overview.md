---
sidebar_position: 1
---

# Overview

Let's check Textgrader **Frontend Project Overview** in 5 minutes.

## Technologies

Textgrader website is a [Next.js](https://nextjs.org/) project, so it uses Next.js concepts that are avaiable with more details [here](https://nextjs.org/docs/getting-started/project-structure). In this page we will cover just what is more relevant for Textgrader use case.

Along with [Next.js](https://nextjs.org/) this project also uses:

1. [Node.js](https://nodejs.org/en) (JavaScript Runtime)
2. [React](https://react.dev/) (JavaScript library for building user interfaces)
3. [Next.js](https://nextjs.org/) (The React Framework for the Web)
4. [Typescript](https://www.typescriptlang.org/) (TypeScript is a strongly typed programming language that builds on JavaScript)

## Running the Project Natively

Besides using [Docker](https://www.docker.com/), it's also possible to run the frontend project natively, for this you will need installed on your machine:

1. [Node.js](https://nodejs.org/en) (Install LTS version)
2. [Yarn](https://yarnpkg.com/)

With this installed you just need to run the command bellow in ``frontend`` folder:

```bash
yarn && yarn dev
```

## File Tree Structure

```txt title="/frontend"
├── README.md                   | Website README file
├── public                      | Website public files folder
│   ├── README.md               | Website README copy for rendering in about page
│   ├── cefet-logo.jpg          | Website institutional logo
│   ├── diagram.png             | Project diagram image
│   ├── favicon.ico             | Website favicon image
│   ├── next.svg                | Next.js image
│   ├── thirteen.svg            | Next.js image
│   └── vercel.svg              | Next.js image
├── src                         | Website source directory
│   ├── pages                   | Next.js routing directory
│   │   ├── _app.tsx            | Next.js server side app wrapper file
│   │   ├── _document.tsx       | Next.js server side document file
│   │   ├── api                 | Unused folder
│   │   │   ├── hello.ts        | Unused file
│   │   │   └── text_grader.ts  | Unused file
│   │   ├── index.tsx           | Website home  page -> http://localhost:3000/
│   │   ├── redacao.tsx         | Website essay page -> http://localhost:3000/redacao
│   │   └── sobre.tsx           | Website about page -> http://localhost:3000/sobre
│   └── styles                  | Website styles folder
│       └── globals.css         | Website global style
├── tsconfig.json               | Website typescript configuration
└── yarn.lock                   | Website yarn lock file
```
