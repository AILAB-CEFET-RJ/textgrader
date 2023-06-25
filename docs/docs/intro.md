---
sidebar_position: 1
---

# Tutorial Intro

Let's discover **Textgrader in less than 3 minutes**.

## Getting Started

Get started by **running the project** with [Docker](https://www.docker.com/).

:::info

This project works on Windows, but for better performance we advise running it on Linux.

:::

## Running the Project

The project will be automatically setup in your machine after you run the commands:

```bash
git clone https://github.com/MLRG-CEFET-RJ/textgrader

cd textgrader

git checkout release/v0.1.2

docker compose up -d
```

You can type this commands into Command Prompt, Powershell, Terminal, or any other integrated terminal of your code editor.

The command also installs all necessary dependencies and execute all necessary scripts you need to run Textgrader.

:::tip

You can also run just a sub project of this project by running the commands bellow:

```bash title="Running just the Frontend"
docker compose up frontend -d
```

```bash title="Running just the Backend"
docker compose up backend -d
```

```bash title="Running just the Documentation"
docker compose up docs -d
```

:::

## Viewing the Project

After you finish running the step above, you can checkout the project live on the links bellow:

- Frontend: [http://localhost:3000/](http://localhost:3000/)
- Backend API: [http://localhost:8000/](http://localhost:8000/)
  - API Endpoint Docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Project Docs: [http://localhost:3333/](http://localhost:3333/)

Textgrader uses an AI model on backend capable of giving grades to essays on real time, this process is detailed in the image bellow:

<p align="center">
  <img src="/img/diagram.png" alt="diagram"/>
</p>
