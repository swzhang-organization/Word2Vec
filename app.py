from flask import Flask, request, render_template
import gensim.downloader as api
import numpy as np

app = Flask(__name__)

# Load a small pretrained embedding model
# Downloaded automatically if not present in local cache
model = api.load("glove-wiki-gigaword-50")


def phrase_vector(text):
    """Return the averaged vector for a phrase."""
    words = text.split()
    vectors = [model[w] for w in words if w in model]
    if not vectors:
        return None
    return np.mean(vectors, axis=0)


def compute_similarity(target, items):
    """Compute similarity scores for a list of words/phrases."""
    target_vec = phrase_vector(target)
    results = []
    for item in items:
        item_vec = phrase_vector(item)
        if target_vec is None or item_vec is None:
            sim = None
        else:
            sim = float(np.dot(target_vec, item_vec) /
                        (np.linalg.norm(target_vec) * np.linalg.norm(item_vec)))
        results.append((item, sim))
    return results


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        target = request.form.get('target', '')
        items_text = request.form.get('items', '')
        items = [line.strip() for line in items_text.splitlines() if line.strip()]
        results = compute_similarity(target, items)
        return render_template('index.html', results=results,
                               target=target, items_text=items_text)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
