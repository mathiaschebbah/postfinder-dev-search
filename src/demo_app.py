from flask import Flask, render_template_string, request, jsonify
from client import PostfinderClient

app = Flask(__name__)

# Use local FastAPI server
client = PostfinderClient(base_url="http://localhost:8000")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Postfinder Embedding Demo</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        h1 { color: #333; }
        .card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .card h2 { margin-top: 0; color: #555; }
        label { display: block; margin-bottom: 5px; font-weight: 500; }
        input, textarea {
            width: 100%;
            padding: 10px;
            margin-bottom: 15px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }
        textarea { min-height: 80px; resize: vertical; }
        button {
            background: #007bff;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }
        button:hover { background: #0056b3; }
        button:disabled { background: #ccc; cursor: not-allowed; }
        .result {
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 4px;
            padding: 15px;
            margin-top: 15px;
            display: none;
        }
        .result.show { display: block; }
        .result pre {
            margin: 0;
            white-space: pre-wrap;
            word-break: break-all;
            font-size: 12px;
            max-height: 200px;
            overflow-y: auto;
        }
        .stats { color: #666; font-size: 14px; margin-bottom: 10px; }
        .similarity-section { margin-top: 20px; padding-top: 20px; border-top: 1px solid #eee; }
        .similarity-score {
            font-size: 48px;
            font-weight: bold;
            text-align: center;
            padding: 20px;
        }
        .similarity-score.high { color: #28a745; }
        .similarity-score.medium { color: #ffc107; }
        .similarity-score.low { color: #dc3545; }
        .loading { opacity: 0.6; }
        .error { color: #dc3545; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        @media (max-width: 600px) { .grid { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
    <h1>Postfinder Embedding Demo</h1>

    <div class="card">
        <h2>API Status</h2>
        <div id="health-status">Checking...</div>
    </div>

    <div class="grid">
        <div class="card">
            <h2>Embed Search Query</h2>
            <form id="query-form">
                <label for="query">Search Query</label>
                <input type="text" id="query" name="query" placeholder="e.g., woman with dog on beach" required>
                <button type="submit">Generate Embedding</button>
            </form>
            <div id="query-result" class="result">
                <div class="stats"></div>
                <pre></pre>
            </div>
        </div>

        <div class="card">
            <h2>Embed Instagram Post</h2>
            <form id="post-form">
                <label for="caption">Caption</label>
                <textarea id="caption" name="caption" placeholder="e.g., Beautiful sunset at the beach! #summer #vibes"></textarea>
                <label for="image_url">Image URL (optional)</label>
                <input type="url" id="image_url" name="image_url" placeholder="https://example.com/image.jpg">
                <button type="submit">Generate Embedding</button>
            </form>
            <div id="post-result" class="result">
                <div class="stats"></div>
                <pre></pre>
            </div>
        </div>
    </div>

    <div class="card similarity-section">
        <h2>Compare Similarity</h2>
        <p>Generate both a query and post embedding above, then compare their similarity.</p>
        <button id="compare-btn" disabled>Calculate Similarity</button>
        <div id="similarity-result" class="result">
            <div class="similarity-score"></div>
        </div>
    </div>

    <script>
        let queryEmbedding = null;
        let postEmbedding = null;

        async function checkHealth() {
            try {
                const res = await fetch('/api/health');
                const data = await res.json();
                document.getElementById('health-status').innerHTML =
                    `<span style="color: ${data.status === 'ok' ? 'green' : 'red'}">
                        ${data.status === 'ok' ? '✓ Connected' : '✗ Error'} - Model: ${data.model}
                    </span>`;
            } catch (e) {
                document.getElementById('health-status').innerHTML =
                    '<span class="error">✗ Cannot connect to API</span>';
            }
        }

        document.getElementById('query-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const btn = e.target.querySelector('button');
            const resultDiv = document.getElementById('query-result');
            btn.disabled = true;
            btn.textContent = 'Generating...';
            resultDiv.classList.remove('show');

            try {
                const res = await fetch('/api/embed/query', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({query: document.getElementById('query').value})
                });
                const data = await res.json();
                if (data.error) throw new Error(data.error);

                queryEmbedding = data.embedding;
                resultDiv.querySelector('.stats').textContent = `Dimension: ${data.dimension}`;
                resultDiv.querySelector('pre').textContent = JSON.stringify(data.embedding.slice(0, 20), null, 2) + '\\n... (truncated)';
                resultDiv.classList.add('show');
                updateCompareButton();
            } catch (err) {
                resultDiv.querySelector('.stats').textContent = '';
                resultDiv.querySelector('pre').innerHTML = `<span class="error">Error: ${err.message}</span>`;
                resultDiv.classList.add('show');
            } finally {
                btn.disabled = false;
                btn.textContent = 'Generate Embedding';
            }
        });

        document.getElementById('post-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const btn = e.target.querySelector('button');
            const resultDiv = document.getElementById('post-result');
            btn.disabled = true;
            btn.textContent = 'Generating...';
            resultDiv.classList.remove('show');

            try {
                const res = await fetch('/api/embed/post', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        caption: document.getElementById('caption').value || null,
                        image_url: document.getElementById('image_url').value || null
                    })
                });
                const data = await res.json();
                if (data.error) throw new Error(data.error);

                postEmbedding = data.embedding;
                resultDiv.querySelector('.stats').textContent = `Dimension: ${data.dimension}`;
                resultDiv.querySelector('pre').textContent = JSON.stringify(data.embedding.slice(0, 20), null, 2) + '\\n... (truncated)';
                resultDiv.classList.add('show');
                updateCompareButton();
            } catch (err) {
                resultDiv.querySelector('.stats').textContent = '';
                resultDiv.querySelector('pre').innerHTML = `<span class="error">Error: ${err.message}</span>`;
                resultDiv.classList.add('show');
            } finally {
                btn.disabled = false;
                btn.textContent = 'Generate Embedding';
            }
        });

        function updateCompareButton() {
            document.getElementById('compare-btn').disabled = !(queryEmbedding && postEmbedding);
        }

        document.getElementById('compare-btn').addEventListener('click', async () => {
            const btn = document.getElementById('compare-btn');
            const resultDiv = document.getElementById('similarity-result');
            btn.disabled = true;
            btn.textContent = 'Calculating...';

            try {
                const res = await fetch('/api/similarity', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({embedding1: queryEmbedding, embedding2: postEmbedding})
                });
                const data = await res.json();

                const score = data.similarity;
                const scoreDiv = resultDiv.querySelector('.similarity-score');
                scoreDiv.textContent = (score * 100).toFixed(1) + '%';
                scoreDiv.className = 'similarity-score ' + (score > 0.7 ? 'high' : score > 0.4 ? 'medium' : 'low');
                resultDiv.classList.add('show');
            } catch (err) {
                resultDiv.querySelector('.similarity-score').innerHTML = `<span class="error">Error: ${err.message}</span>`;
                resultDiv.classList.add('show');
            } finally {
                btn.disabled = false;
                btn.textContent = 'Calculate Similarity';
            }
        });

        checkHealth();
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/health")
def health():
    try:
        return jsonify(client.health())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/embed/query", methods=["POST"])
def embed_query():
    try:
        data = request.json
        result = client.embed_query(data["query"])
        return jsonify({"embedding": result.embedding, "dimension": result.dimension})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/embed/post", methods=["POST"])
def embed_post():
    try:
        data = request.json
        if not data.get("caption") and not data.get("image_url") and not data.get("image_urls"):
            return jsonify({"error": "At least caption, image_url, or image_urls required"}), 400
        result = client.embed_post(
            caption=data.get("caption"),
            image_url=data.get("image_url"),
            image_urls=data.get("image_urls"),
            dimension=data.get("dimension"),
        )
        return jsonify({"embedding": result.embedding, "dimension": result.dimension})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/embed/posts", methods=["POST"])
def embed_posts():
    try:
        data = request.json
        if not data.get("posts"):
            return jsonify({"error": "posts list required"}), 400
        results = client.embed_posts(
            posts=data["posts"],
            dimension=data.get("dimension"),
        )
        return jsonify({
            "embeddings": [r.embedding for r in results],
            "dimension": results[0].dimension if results else 0,
            "count": len(results),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/similarity", methods=["POST"])
def similarity():
    try:
        data = request.json
        score = client.similarity(data["embedding1"], data["embedding2"])
        return jsonify({"similarity": score})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
