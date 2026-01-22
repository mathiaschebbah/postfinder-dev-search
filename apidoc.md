# Postfinder Embedding API

API de génération d'embeddings multimodaux pour posts Instagram (texte + images).

**Base URL:** `http://localhost:8000`

**Modèle:** Qwen3-VL-Embedding-2B (2048 dimensions, support MRL)

---

## Endpoints

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/embed_query` | Embedding d'une requête de recherche |
| POST | `/embed_post` | Embedding d'un post (caption + images) |
| POST | `/embed_posts` | Embedding batch de plusieurs posts |

---

## GET /health

Vérifie que l'API est opérationnelle.

### Requête

```bash
curl http://localhost:8000/health
```

### Réponse

```json
{
  "status": "ok",
  "model": "Qwen/Qwen3-VL-Embedding-2B"
}
```

---

## POST /embed_query

Génère un embedding pour une requête de recherche textuelle.

### Requête

```bash
curl -X POST http://localhost:8000/embed_query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "femme avec chien sur la plage",
    "dimension": 256
  }'
```

### Paramètres

| Champ | Type | Requis | Description |
|-------|------|--------|-------------|
| `query` | string | oui | Texte de la requête de recherche |
| `dimension` | int | non | Dimension MRL : 64, 128, 256, 512, 1024, 2048. Par défaut : 2048 |

### Réponse

```json
{
  "embedding": [0.0123, -0.0456, 0.0789, ...],
  "dimension": 256
}
```

---

## POST /embed_post

Génère un embedding pour un post Instagram (caption et/ou images).

### Requête

```bash
curl -X POST http://localhost:8000/embed_post \
  -H "Content-Type: application/json" \
  -d '{
    "post": {
      "caption": "Magnifique coucher de soleil sur la plage! #summer #vibes",
      "image_urls": [
        "https://example.com/image1.jpg",
        "https://example.com/image2.jpg"
      ]
    },
    "dimension": 256
  }'
```

### Paramètres

| Champ | Type | Requis | Description |
|-------|------|--------|-------------|
| `post` | object | oui | Objet Post |
| `post.caption` | string | non | Texte du post |
| `post.image_urls` | string[] | non | Liste d'URLs d'images |
| `dimension` | int | non | Dimension MRL : 64, 128, 256, 512, 1024, 2048. Par défaut : 2048 |

> **Note:** Au moins `caption` ou `image_urls` doit être fourni.

### Réponse

```json
{
  "embedding": [0.0123, -0.0456, 0.0789, ...],
  "dimension": 256
}
```

---

## POST /embed_posts

Génère des embeddings pour plusieurs posts en une seule requête (batch).

### Requête

```bash
curl -X POST http://localhost:8000/embed_posts \
  -H "Content-Type: application/json" \
  -d '{
    "posts": [
      {
        "caption": "Premier post",
        "image_urls": ["https://example.com/img1.jpg"]
      },
      {
        "caption": "Deuxième post sans image",
        "image_urls": []
      },
      {
        "caption": null,
        "image_urls": ["https://example.com/img2.jpg", "https://example.com/img3.jpg"]
      }
    ],
    "dimension": 256
  }'
```

### Paramètres

| Champ | Type | Requis | Description |
|-------|------|--------|-------------|
| `posts` | Post[] | oui | Liste d'objets Post |
| `posts[].caption` | string | non | Texte du post |
| `posts[].image_urls` | string[] | non | Liste d'URLs d'images |
| `dimension` | int | non | Dimension MRL commune à tous les embeddings |

### Réponse

```json
{
  "embeddings": [
    [0.0123, -0.0456, ...],
    [0.0234, -0.0567, ...],
    [0.0345, -0.0678, ...]
  ],
  "dimension": 256,
  "count": 3
}
```

---

## Dimensions MRL (Matryoshka Representation Learning)

Le modèle supporte la réduction de dimension via MRL. Les embeddings peuvent être tronqués aux dimensions suivantes sans perte significative de qualité :

| Dimension | Usage recommandé |
|-----------|------------------|
| 64 | Recherche rapide, stockage minimal |
| 128 | Bon compromis vitesse/qualité |
| 256 | Usage général recommandé |
| 512 | Haute précision |
| 1024 | Très haute précision |
| 2048 | Précision maximale (défaut) |

---

## Calcul de similarité

Les embeddings sont normalisés L2. Utilisez le **produit scalaire** (dot product) ou la **similarité cosinus** pour comparer deux embeddings :

```python
import numpy as np

def similarity(embedding1, embedding2):
    a = np.array(embedding1)
    b = np.array(embedding2)
    return float(np.dot(a, b))  # Équivalent à cosine similarity car normalisé
```

**Interprétation des scores :**
- `> 0.7` : Très similaire
- `0.4 - 0.7` : Moyennement similaire
- `< 0.4` : Peu similaire

---

## Codes d'erreur

| Code | Description |
|------|-------------|
| 200 | Succès |
| 400 | Requête invalide (paramètres manquants) |
| 422 | Erreur de validation (format JSON invalide) |
| 500 | Erreur serveur |

### Exemple d'erreur

```json
{
  "detail": [
    {
      "loc": ["body", "query"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

---

## Exemples d'intégration

### Python

```python
import requests

BASE_URL = "http://localhost:8000"

# Embed une requête
response = requests.post(f"{BASE_URL}/embed_query", json={
    "query": "chien sur la plage",
    "dimension": 256
})
query_embedding = response.json()["embedding"]

# Embed un post
response = requests.post(f"{BASE_URL}/embed_post", json={
    "post": {
        "caption": "Mon chien adore la mer!",
        "image_urls": ["https://example.com/dog.jpg"]
    },
    "dimension": 256
})
post_embedding = response.json()["embedding"]

# Calculer la similarité
import numpy as np
similarity = np.dot(query_embedding, post_embedding)
print(f"Similarité: {similarity:.4f}")
```

### JavaScript/Node.js

```javascript
const BASE_URL = "http://localhost:8000";

// Embed une requête
const queryResponse = await fetch(`${BASE_URL}/embed_query`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    query: "chien sur la plage",
    dimension: 256
  })
});
const { embedding: queryEmbedding } = await queryResponse.json();

// Embed un post
const postResponse = await fetch(`${BASE_URL}/embed_post`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    post: {
      caption: "Mon chien adore la mer!",
      image_urls: ["https://example.com/dog.jpg"]
    },
    dimension: 256
  })
});
const { embedding: postEmbedding } = await postResponse.json();

// Calculer la similarité (dot product)
const similarity = queryEmbedding.reduce((sum, val, i) => sum + val * postEmbedding[i], 0);
console.log(`Similarité: ${similarity.toFixed(4)}`);
```

### cURL - Batch processing

```bash
curl -X POST http://localhost:8000/embed_posts \
  -H "Content-Type: application/json" \
  -d '{
    "posts": [
      {"caption": "Post 1", "image_urls": []},
      {"caption": "Post 2", "image_urls": []},
      {"caption": "Post 3", "image_urls": []}
    ],
    "dimension": 256
  }'
```

---

## Limites et performances

| Paramètre | Valeur |
|-----------|--------|
| Timeout requête | 180s (single), 600s (batch) |
| Max images par post | Pas de limite stricte |
| Max tokens contexte | 8192 |
| Taille max image | ~1.8M pixels |

### Recommandations pour le batch

- Traiter par chunks de **50-100 posts** par requête
- Le traitement est séquentiel côté serveur (GPU)
- Pour 19K posts : ~50-100 requêtes batch

---

## Démarrage du serveur

```bash
# Avec uv
uv run uvicorn server_local:app --host 0.0.0.0 --port 8000

# Avec hot-reload (développement)
uv run uvicorn server_local:app --host 0.0.0.0 --port 8000 --reload
```
