# This Dockerfile is used to build the frontend image for the research agent project.
# It installs the necessary dependencies and sets up the environment for the frontend.
FROM node:21-slim

WORKDIR /app

COPY package*.json ./ 
RUN npm install

COPY . .

EXPOSE 3000
CMD ["npm", "run", "dev"]