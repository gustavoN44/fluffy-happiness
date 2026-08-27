# Multi-stage: Node builds the bundle, nginx serves it. The runtime image carries no
# Node and no node_modules — it ships static files plus a reverse proxy.
FROM node:22-alpine AS build
WORKDIR /build
COPY package.json ./
RUN npm install
COPY . .
RUN npm run build

FROM nginx:1.27-alpine
COPY --from=build /build/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
