#### How to configure nginx

- Run docker compose up
- Frontend next routes should match with nginx ones
    - Next.js pages names are the routes. So, to have a /textgrader/redacao you need a textgrader folder and a redacao.tsx file. 
    - Next.js needs a "/" route, which stays at index.tsx, if its necessary duplicates it and change it name to home, for example. 
- Create a file into /etc/nginx/sites-available folder with application name
- Remove link from default one using
```bash
    sudo unlink /etc/nginx/sites-enabled/default
    sudo ln -s /etc/nginx/sites-available/textgrader /etc/nginx/sites-enabled
```
- Add this config to your sites-available new file
```bash
    server {
        listen 80;
        server_name _;

        location /textgrader {
            proxy_read_timeout 300s;
            proxy_connect_timeout 75s;

            proxy_pass http://localhost:3000;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            }

        location ~ /_next {
            proxy_pass http://localhost:3000;
        }

        location = /textgrader-api {
            add_header 'Access-Control-Allow-Origin' '*';
            proxy_pass http://localhost:5000;
        }
    }

```
- After any changes at nginx, check its syntaxe
```bash
    sudo nginx -t
```
- Restart nginx and check its status
```bash
    sudo service nginx restart
    sudo service nginx status
```