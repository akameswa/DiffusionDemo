"""
Auxiliary Content for Diffusion Demo

Serves the content Gradio can't easily handle directly, namely the interactive
3D embedding plots (and their example images), from a small Flask app.

The Flask app is mounted on the main Gradio server at ``/aux`` instead of being
run as a separate server behind a gradio.live tunnel. That makes the plot
iframes same-origin, relative URLs (``/aux/plot/<session_hash>``), so they keep
working across pod restarts, failovers and domain changes — there is no tunnel
URL to go stale, and nothing extra to expose through the ingress.
"""

from flask import Flask, send_file
from starlette.middleware.wsgi import WSGIMiddleware

from src.util.base import get_user_dir, get_user_examples_dir

# Initialize the Flask application
flask_app = Flask(__name__)

# Path prefix, on the same server/origin as Gradio, under which this app is mounted
AUX_PATH_PREFIX = "/aux"


@flask_app.route("/plot/<session_hash>")
def serve_user_plot(session_hash):
    """
    Serve a user's interactive embedding plot HTML file.

    This endpoint retrieves and serves the 3D visualization of word embeddings
    that was generated for a specific user session. The plot is an HTML file
    created with Plotly that contains JavaScript for interactive visualization.

    Args:
        session_hash (str): Unique identifier for the user session

    Returns:
        HTML file response, or a small page that waits for the plot and then
        reloads the demo (e.g. a tab left open across a pod restart).
    """
    # Get the directory for this user's session
    user_dir = get_user_dir(session_hash)
    if not user_dir:
        return _plot_loading_page()

    # Construct the path to the HTML plot file
    html_path = user_dir / "embedding_plot.html"

    # Serve the file if it exists, otherwise wait for it
    if html_path.exists():
        try:
            return send_file(str(html_path.absolute()), mimetype="text/html")
        except Exception as e:
            print(f"Error serving file: {e}")
            return f"Error serving file: {e}", 500

    return _plot_loading_page()


@flask_app.route("/examples/<session_hash>/<image_name>")
def serve_user_example_image(session_hash, image_name):
    """
    Serve an example image from a user's session directory.

    Args:
        session_hash (str): Unique identifier for the user session
        image_name (str): Name of the image file to serve

    Returns:
        Image file response or error message
    """
    # Get the examples directory for this user's session
    examples_dir = get_user_examples_dir(session_hash)
    if not examples_dir:
        return "Invalid session", 404

    # Construct the path to the image file
    image_path = examples_dir / image_name

    # Serve the image if it exists, otherwise return an error
    if image_path.exists():
        try:
            return send_file(str(image_path.absolute()), mimetype="image/jpeg")
        except Exception as e:
            print(f"Error serving user example image: {e}")
            return f"Error serving image: {e}", 500

    return f"Image not found at {image_path.absolute()}", 404


def _plot_loading_page():
    """
    Placeholder for a plot that isn't available (yet).

    Returns 404 on purpose: the page polls the same URL and reloads itself once
    the plot exists (200). If the session is really gone (pod restart or
    failover), it reloads the whole demo to get a fresh session — which is safe
    and possible only because the iframe is same-origin with the demo now.
    """
    html = """<!doctype html>
<html><head><meta charset="utf-8"><title>Embedding plot</title></head>
<body style="font-family:sans-serif;display:flex;align-items:center;justify-content:center;height:100vh;margin:0">
<div style="text-align:center;color:#555">Preparing the embedding plot&hellip;</div>
<script>
let tries = 0;
const timer = setInterval(() => {
  if (++tries > 10) { clearInterval(timer); window.top.location.href = "/"; return; }
  fetch(location.href, { cache: "no-store" })
    .then(r => { if (r.ok) { clearInterval(timer); location.reload(); } })
    .catch(() => {});
}, 3000);
</script>
</body></html>"""
    return html, 404


def mount_aux_server(demo):
    """
    Mount the Flask app on the Gradio server, under AUX_PATH_PREFIX.

    Must be called after demo.launch(), because launch() rebuilds demo.app.

    Args:
        demo (gr.Blocks): The main Gradio Blocks app.
    """
    demo.app.mount(AUX_PATH_PREFIX, WSGIMiddleware(flask_app))


def get_aux_base_url():
    """
    Get the base URL prefix under which the auxiliary content is served.

    Returns:
        str: Relative path prefix (e.g. "/aux"), so URLs inherit whatever
             origin the browser used to load the demo.
    """
    return AUX_PATH_PREFIX
