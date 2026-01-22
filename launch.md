# Lancement du serveur sur Mac Studio

## Prérequis

- Mac Studio avec Apple Silicon
- Python 3.12+
- `uv` installé
- `tmux` installé (`brew install tmux`)

## Démarrer le serveur

```bash
ssh mathias@<ip-macstudio>
cd ~/Projects/postfinder-dev-search
tmux new -s postfinder
uv run uvicorn server_local:app --host 0.0.0.0 --port 8933
```

Puis **Ctrl+B** puis **D** pour détacher et quitter SSH.

## Commandes utiles

| Action | Commande |
|--------|----------|
| Voir les sessions | `tmux ls` |
| Se rattacher | `tmux attach -t postfinder` |
| Détacher | `Ctrl+B` puis `D` |
| Arrêter le serveur | `tmux attach -t postfinder` puis `Ctrl+C` |
| Tuer la session | `tmux kill-session -t postfinder` |

## Vérifier le status

```bash
curl http://<ip-macstudio>:8933/health
```

Réponse attendue :
```json
{"status":"ok","model":"Qwen/Qwen3-VL-Embedding-2B"}
```

## Logs

Les logs s'affichent dans la session tmux. Pour les voir :
```bash
tmux attach -t postfinder
```

## URL de l'API

```
http://<ip-macstudio>:8933
```

Endpoints :
- `GET /health`
- `POST /embed_query`
- `POST /embed_post`
- `POST /embed_posts`
