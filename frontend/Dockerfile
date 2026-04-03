# Stage 1 — Build
FROM node:22-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
ARG VITE_API_BASE=http://localhost:8000
ARG VITE_DEMO_TOKEN=demo-token
ENV VITE_API_BASE=$VITE_API_BASE
ENV VITE_DEMO_TOKEN=$VITE_DEMO_TOKEN
RUN npm run build

# Stage 2 — Serve via Nginx
FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
# SPA routing: redirect all to index.html
RUN echo 'server { listen 80; location / { root /usr/share/nginx/html; try_files $uri /index.html; } }' \
  > /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
