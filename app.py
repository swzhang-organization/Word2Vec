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


def compute_similarity_matrix(targets, items):
    """Compute similarity matrix for multiple targets and items."""
    target_vecs = [phrase_vector(t) for t in targets]
    item_vecs = [phrase_vector(i) for i in items]

    matrix = []
    for tvec in target_vecs:
        row = []
        for ivec in item_vecs:
            if tvec is None or ivec is None:
                sim = None
            else:
                sim = float(np.dot(tvec, ivec) /
                            (np.linalg.norm(tvec) * np.linalg.norm(ivec)))
            row.append(sim)
        matrix.append(row)
    return matrix


@app.template_filter('sim_color')
def similarity_to_color(sim):
    """Return a color hex based on similarity for green gradient."""
    if sim is None:
        return '#ffffff'
    val = max(0.0, min(1.0, sim))
    r = int(255 * (1 - val))
    g = 255
    b = int(255 * (1 - val))
    return f'#{r:02x}{g:02x}{b:02x}'


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        targets_text = request.form.get('targets', '')
        items_text = request.form.get('items', '')
        targets = [line.strip() for line in targets_text.splitlines() if line.strip()]
        items = [line.strip() for line in items_text.splitlines() if line.strip()]
        matrix = compute_similarity_matrix(targets, items)
        return render_template('index.html', results=matrix,
                               targets=targets, items=items,
                               targets_text=targets_text, items_text=items_text)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
