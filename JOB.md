# Cloud Run Job - Embedding Generator

Job pour générer les embeddings de tous les posts Instagram.

## Architecture

```
postfinder-dev-search/
├── src/
│   ├── client.py              # Client API embedding (inclus dans le job)
│   └── embeddings/
│       ├── generate.py        # Script principal
│       ├── models.py          # Modèles SQLModel
│       ├── Dockerfile         # Image Docker du job
│       └── requirements.txt   # Dépendances minimales (sans torch)
```

Seuls `src/client.py` et `src/embeddings/` sont déployés. Le reste du projet (serveur FastAPI, torch, transformers) n'est pas inclus.

## Image Docker

**Registry:** `europe-west1-docker.pkg.dev/postfinder-dev-v2/postfinder-dev-embeddings/embeddings`

**Taille:** ~360 MB

### Build et push

```bash
cd postfinder-dev-search

# Build pour linux/amd64 (Cloud Run) et push
docker buildx build --platform linux/amd64 \
  -f src/embeddings/Dockerfile \
  -t europe-west1-docker.pkg.dev/postfinder-dev-v2/postfinder-dev-embeddings/embeddings \
  --push .
```

## Variables d'environnement

| Variable | Description | Défaut |
|----------|-------------|--------|
| `DATABASE_URL` | URL PostgreSQL (secret) | **requis** |
| `EMBEDDING_API_URL` | URL de l'API d'embedding | `https://embedding.views.fr` |
| `EMBEDDING_DIMENSION` | Dimension des vecteurs | `512` |
| `BATCH_SIZE` | Posts par appel API | `10` |

Variables injectées automatiquement par Cloud Run :
- `CLOUD_RUN_TASK_INDEX` : Index de la tâche (0, 1, 2...)
- `CLOUD_RUN_TASK_COUNT` : Nombre total de tâches

## Parallélisme

Le job utilise le partitionnement par modulo sur `ig_media_id` pour distribuer les posts entre les tâches.

**Important** : On partitionne par `ig_media_id` (stable) et non par index de liste (instable), car les tâches démarrent à des moments différents et voient des listes différentes.

Code :
```python
for row in all_rows:
    if int(row.ig_media_id) % task_count == task_index:
        posts.append(...)
```

## Commandes gcloud

### Créer le job

```bash
gcloud run jobs create postfinder-embeddings \
  --region=europe-west1 \
  --image=europe-west1-docker.pkg.dev/postfinder-dev-v2/postfinder-dev-embeddings/embeddings \
  --tasks=4 \
  --task-timeout=3600s \
  --memory=512Mi \
  --set-secrets="DATABASE_URL=DATABASE_URL:latest"
```

### Exécuter le job

```bash
gcloud run jobs execute postfinder-embeddings --region=europe-west1
```

### Exécuter avec un parallélisme différent

```bash
gcloud run jobs execute postfinder-embeddings --region=europe-west1 --tasks=8
```

### Voir les logs

```bash
gcloud run jobs executions list --job=postfinder-embeddings --region=europe-west1
```
