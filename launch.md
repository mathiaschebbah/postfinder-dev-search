# Lancement du serveur sur Mac Studio

## Prérequis

- Mac Studio avec Apple Silicon
- Python 3.12+
- uv installé
- tmux installé (brew install tmux)
- cloudflared installé (brew install cloudflared)

## Connexion SSH

```
ssh mathias@macstudio
```

ou

```
ssh alexmouchet@192.168.21.54
```

## Lancer le serveur (session tmux "postfinder")

```
tmux new -s postfinder
```

```
cd ~/Projects/postfinder-dev-search
uv run uvicorn server_local:app --host 0.0.0.0 --port 8933
```

Détacher : Ctrl+B puis D

## Lancer le tunnel Cloudflare (session tmux "tunnel")

```
tmux new -s tunnel
```

```
cloudflared tunnel --url http://localhost:8933
```

Détacher : Ctrl+B puis D

L'URL publique s'affiche dans le terminal (ex: https://xxx.trycloudflare.com)

## Commandes tmux

Lister les sessions :
```
tmux ls
```

Se rattacher à une session :
```
tmux attach -t postfinder
tmux attach -t tunnel
```

Détacher (depuis une session) :
```
Ctrl+B puis D
```

Changer de session (depuis une session) :
```
Ctrl+B puis s
```

Tuer une session :
```
tmux kill-session -t postfinder
tmux kill-session -t tunnel
```

## Vérifier le status

Local :
```
curl http://localhost:8933/health
```

Via tunnel :
```
curl https://xxx.trycloudflare.com/health
```

Réponse attendue :
```
{"status":"ok","model":"Qwen/Qwen3-VL-Embedding-2B"}
```

## Arrêter le serveur

```
tmux attach -t postfinder
Ctrl+C
exit
```

## Arrêter le tunnel

```
tmux attach -t tunnel
Ctrl+C
exit
```

## Endpoints API

- GET /health
- POST /embed_query
- POST /embed_post
- POST /embed_posts
