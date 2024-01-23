# Use a imagem Node.js
FROM node:lts-alpine

# Defina o diretório de trabalho
WORKDIR /app

# Copie os arquivos do projeto para o contêiner
COPY /frontend /app

# Instale as dependências
RUN npm install

# Exponha a porta do aplicativo Next.js
EXPOSE 3000

# Comando para iniciar o aplicativo
CMD ["npm", "run", "dev"]
