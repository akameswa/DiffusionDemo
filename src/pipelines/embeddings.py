"""
CLIP embeddings visualization pipeline for the Diffusion Demo application.

This module provides functionality to visualize and explore the CLIP embedding space,
allowing users to see how different words and concepts relate to each other in the
semantic space used by the diffusion model. It includes interactive 3D visualization,
word embedding generation, and image generation for concepts.
"""

import io
import os
import random
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt

import base64
from io import BytesIO
import plotly.express as px
from PIL import Image

from src.util.base import (
    get_word_embeddings,
    get_concat_embeddings,
    calculate_residual,
    get_axis_embeddings,
    get_user_dir,
    get_user_examples_dir,
    get_user_viz_dir,
)
from src.util.params import num_inference_steps, guidance_scale, tokenizer, pipe, negative_prompt
from src.util.clip_config import (
    masculine,
    feminine,
    young,
    old,
    common,
    elite,
    examples,
    axis_names,
    axisMap,
)
from PIL import Image
import time

from serve import get_aux_base_url

# Initialize the semantic axes from the word lists
age = get_axis_embeddings(young, old)
gender = get_axis_embeddings(masculine, feminine)
royalty = get_axis_embeddings(common, elite)


def generate_examples(
    examples=examples,
    num_inference_steps=num_inference_steps,
    guidance_scale=guidance_scale,
):
    """
    Generate images for a list of example words.
    
    Args:
        examples (list): List of words to generate images for
        num_inference_steps (int): Number of denoising steps
        guidance_scale (float): Text guidance scale
        
    Returns:
        list: Base64-encoded images as data URLs
    """
    images = []
    for example in examples:
        # Generate image using the diffusion pipeline
        image = pipe(
            prompt=example,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]
        
        # Convert to base64 for web display
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        images.append("data:image/jpeg;base64, " + encoded_image)
    return images


# Generate default examples for initial visualization
images = generate_examples()

# Set up the embedding space with three principal axes
axis = np.vstack([gender, royalty, age])
# Calculate the residual axis (orthogonal to the other two)
axis[1] = calculate_residual(axis, axis_names)

# Project example words onto the axes to get 3D coordinates
coords = get_concat_embeddings(examples) @ axis.T
coords[:, 1] = 5 * (1.0 - coords[:, 1])  # Scale the residual axis for better visualization

# Store default values to initialize new user sessions
default_examples = examples.copy()
default_images = images.copy()
default_coords = coords.copy()
user_data = {}  # Dictionary to store per-user session data


def get_safe_filename(word):
    """
    Convert a word to a safe filename by removing special characters.
    
    Args:
        word (str): Input word or phrase
        
    Returns:
        str: Safe filename version of the input
    """
    return "".join([c if c.isalnum() else "_" for c in word])


def generate_user_html(session_hash):
    """
    Generate the HTML file for a user's 3D embedding visualization.
    
    Args:
        session_hash (str): Unique identifier for the user session
        
    Returns:
        Path: Path to the generated HTML file, or None if generation failed
    """
    if not session_hash:
        return None

    user_dir = get_user_dir(session_hash)
    if not user_dir:
        return None

    html_path = user_dir / "embedding_plot.html"
    abs_html_path = html_path.absolute()
    str_html_path = str(abs_html_path)

    # Get the user's 3D figure and write it to an HTML file
    user_fig = user_data[session_hash]["fig"]

    user_fig.write_html(
        str_html_path,
        full_html=True,
        include_plotlyjs="cdn",
        config={"responsive": True},
    )

    print(f"Generated HTML at: {str_html_path}")
    print(f"File exists after generation: {html_path.exists()}")
    print(f"Absolute path: {abs_html_path}")

    # Make the file readable by the web server
    try:
        os.chmod(str_html_path, 0o644)
    except Exception as e:
        print(f"Warning: Could not set file permissions: {e}")

    return html_path


def is_new_session(session_hash):
    """
    Check if this is a new user session that needs initialization.
    
    Args:
        session_hash (str): Unique identifier for the user session
        
    Returns:
        bool: True if this is a new session, False otherwise
    """
    return session_hash not in user_data


def init_user_session(request: gr.Request):
    """
    Initialize a user session with default examples and visualization.
    
    This function sets up the 3D embedding visualization for a new user,
    either loading pre-generated images or generating new ones as needed.
    
    Args:
        request (gr.Request): Gradio request object containing session information
        
    Returns:
        tuple: (
            str: URL to the embedding visualization, 
            str: Session hash, 
            bool: Whether this was a new session
        )
    """
    session_hash = request.session_hash
    if not session_hash:
        session_hash = str(random.randint(10000, 99999))

    print(f"Initializing session for: {session_hash}")

    is_new = is_new_session(session_hash)

    if is_new:
        # Initialize new user data with default values
        user_data[session_hash] = {
            "examples": default_examples.copy(),
            "images": {},  
            "coords": default_coords.copy(),
            "axis": axis.copy(),
            "axis_names": axis_names.copy(),
        }

        # Try to load pre-generated examples from disk
        base_examples_dir = "DiffusionDemo/images/examples"
        
        for example in user_data[session_hash]["examples"]:
            safe_filename = get_safe_filename(example)
            source_path = os.path.join(base_examples_dir, f"{safe_filename}.jpg")
            
            if os.path.exists(source_path):
                try:
                    # Load pre-generated image from disk
                    with open(source_path, 'rb') as f:
                        img_data = f.read()
                    img_str = base64.b64encode(img_data).decode('utf-8')
                    user_data[session_hash]["images"][example] = img_str
                    print(f"Loaded pre-generated image for '{example}'")
                except Exception as e:
                    print(f"Error loading pre-generated image for '{example}': {e}")
                    try:
                        # Generate image if loading failed
                        image = pipe(
                            prompt=example,
                            negative_prompt=negative_prompt,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                        ).images[0]
                        user_data[session_hash]["images"][example] = image_to_base64(image)
                        print(f"Generated fallback image for '{example}'")
                    except Exception as e2:
                        print(f"Error generating fallback image for '{example}': {e2}")
            else:
                try:
                    # No pre-generated image, generate a new one
                    print(f"No pre-generated image found for '{example}', generating one...")
                    image = pipe(
                        prompt=example,
                        negative_prompt=negative_prompt,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                    ).images[0]
                    user_data[session_hash]["images"][example] = image_to_base64(image)
                except Exception as e:
                    print(f"Error generating image for '{example}': {e}")

        # Create the 3D scatter plot visualization
        user_fig = px.scatter_3d(
            x=user_data[session_hash]["coords"][:, 0],
            y=user_data[session_hash]["coords"][:, 1],
            z=user_data[session_hash]["coords"][:, 2],
            labels={
                "x": user_data[session_hash]["axis_names"][0],
                "y": user_data[session_hash]["axis_names"][1],
                "z": user_data[session_hash]["axis_names"][2],
            },
            text=user_data[session_hash]["examples"],
            height=750,
        )

        # Configure the 3D visualization layout
        user_fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=0),
            scene_camera=dict(eye=dict(x=2, y=2, z=0.1)),
        )

        # Set up hover behavior for the points
        user_fig.update_traces(
            hoverinfo="text+x+y+z",
            hovertemplate="%{x:.2f}, %{y:.2f}, %{z:.2f}"
        )
        
        user_data[session_hash]["fig"] = user_fig

    elif "coords" not in user_data[session_hash] or len(user_data[session_hash]["coords"]) != len(user_data[session_hash]["examples"]):
        # Update coordinates if examples changed but session exists
        user_data[session_hash]["coords"] = (
            get_concat_embeddings(user_data[session_hash]["examples"])
            @ user_data[session_hash]["axis"].T
        )
        user_data[session_hash]["coords"][:, 1] = 5 * (
            1.0 - user_data[session_hash]["coords"][:, 1]
        )
        
        # Create or update the visualization if needed
        if "fig" not in user_data[session_hash]:
            user_fig = px.scatter_3d(
                x=user_data[session_hash]["coords"][:, 0],
                y=user_data[session_hash]["coords"][:, 1],
                z=user_data[session_hash]["coords"][:, 2],
                labels={
                    "x": user_data[session_hash]["axis_names"][0],
                    "y": user_data[session_hash]["axis_names"][1],
                    "z": user_data[session_hash]["axis_names"][2],
                },
                text=user_data[session_hash]["examples"],
                height=750,
            )

            user_fig.update_layout(
                margin=dict(l=0, r=0, b=0, t=0),
                scene_camera=dict(eye=dict(x=2, y=2, z=0.1)),
            )

            user_fig.update_traces(
                hoverinfo="text+x+y+z",
                hovertemplate="%{x:.2f}, %{y:.2f}, %{z:.2f}"
            )
            
            user_data[session_hash]["fig"] = user_fig

    # Generate HTML for the visualization and get the URL
    html_path = generate_user_html(session_hash)
    timestamp = int(time.time())
    flask_url = f"{get_aux_base_url()}/plot/{session_hash}?t={timestamp}"

    return flask_url, session_hash, is_new


def update_user_fig(session_hash):
    """
    Update the 3D visualization for a user session.
    
    Args:
        session_hash (str): Unique identifier for the user session
        
    Returns:
        str: URL to the updated visualization
    """
    # Update the coordinates and labels in the figure
    user_data[session_hash]["fig"].data[0].x = user_data[session_hash]["coords"][:, 0]
    user_data[session_hash]["fig"].data[0].y = user_data[session_hash]["coords"][:, 1]
    user_data[session_hash]["fig"].data[0].z = user_data[session_hash]["coords"][:, 2]
    user_data[session_hash]["fig"].data[0].text = user_data[session_hash]["examples"]

    # Update axis labels
    user_data[session_hash]["fig"].update_layout(
        scene=dict(
            xaxis_title=user_data[session_hash]["axis_names"][0],
            yaxis_title=user_data[session_hash]["axis_names"][1],
            zaxis_title=user_data[session_hash]["axis_names"][2],
        )
    )
    
    # Update hover behavior
    user_data[session_hash]["fig"].update_traces(
        hoverinfo="text+x+y+z",
        hovertemplate="%{x:.2f}, %{y:.2f}, %{z:.2f}"
    )

    # Generate updated HTML and return the URL
    html_path = generate_user_html(session_hash)

    timestamp = int(time.time())
    return f"{get_aux_base_url()}/plot/{session_hash}?t={timestamp}"

def add_word_user(new_example, session_hash):
    """
    Add a new word to the user's visualization.
    
    Args:
        new_example (str): Word to add
        session_hash (str): Unique identifier for the user session
        
    Returns:
        str: URL to the updated visualization
    """
    user_examples = user_data[session_hash]["examples"]
    user_coords = user_data[session_hash]["coords"]
    user_axis = user_data[session_hash]["axis"]

    # Calculate new word's coordinates in semantic space
    new_coord = get_concat_embeddings([new_example]) @ user_axis.T
    new_coord[:, 1] = 5 * (1.0 - new_coord[:, 1])
    user_data[session_hash]["coords"] = np.vstack([user_coords, new_coord])

    # Generate image for the new word
    image = pipe(
        prompt=new_example,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).images[0]

    # Store the image and update examples list
    if "images" not in user_data[session_hash]:
        user_data[session_hash]["images"] = {}
    
    user_data[session_hash]["images"][new_example] = image_to_base64(image)
    user_data[session_hash]["examples"].append(new_example)

    return update_user_fig(session_hash)


def remove_word_user(word_to_remove, session_hash):
    """
    Remove a word from the user's visualization.
    
    Args:
        word_to_remove (str): Word to remove
        session_hash (str): Unique identifier for the user session
        
    Returns:
        str: URL to the updated visualization
    """
    user_examples = user_data[session_hash]["examples"]
    user_coords = user_data[session_hash]["coords"]

    # Find the word in the examples list
    examplesMap = {example: index for index, example in enumerate(user_examples)}
    if word_to_remove not in examplesMap:
        return update_user_fig(session_hash)

    index = examplesMap[word_to_remove]

    # Remove image and update coordinates and examples
    if "images" in user_data[session_hash]:
        user_data[session_hash]["images"].pop(word_to_remove, None)

    user_data[session_hash]["coords"] = np.delete(user_coords, index, 0)
    user_data[session_hash]["examples"].pop(index)

    return update_user_fig(session_hash)


def add_rem_word_user(new_examples, session_hash):
    """
    Add or remove multiple words from the user's visualization.
    
    If a word is already present, it will be removed. If not, it will be added.
    
    Args:
        new_examples (str): Space or comma separated list of words
        session_hash (str): Unique identifier for the user session
        
    Returns:
        str: URL to the updated visualization
    """
    new_examples = new_examples.replace(",", " ").split()

    for new_example in new_examples:
        if new_example in user_data[session_hash]["examples"]:
            # Remove if already present
            remove_word_user(new_example, session_hash)
            gr.Info(f"Removed {new_example}")
        else:
            # Check if word exists in vocabulary and add if valid
            tokens = tokenizer.encode(new_example)
            if len(tokens) != 3:
                gr.Warning(f"{new_example} not found in embeddings")
            else:
                add_word_user(new_example, session_hash)
                gr.Info(f"Added {new_example}")

    return update_user_fig(session_hash)


def change_word_user(examples, session_hash):
    """
    Regenerate images for existing words in the visualization.
    
    Args:
        examples (str): Space or comma separated list of words
        session_hash (str): Unique identifier for the user session
        
    Returns:
        str: URL to the updated visualization
    """
    examples = examples.replace(",", " ").split()

    for example in examples:
        if example in user_data[session_hash]["examples"]:
            # Remove and re-add to regenerate the image
            remove_word_user(example, session_hash)
            add_word_user(example, session_hash)
            gr.Info(f"Changed image for {example}")

    return update_user_fig(session_hash)


def clear_words_user(session_hash):
    """
    Clear all words from the user's visualization.
    
    Args:
        session_hash (str): Unique identifier for the user session
        
    Returns:
        str: URL to the updated visualization or empty string on error
    """
    if session_hash in user_data:
        while user_data[session_hash]["examples"]:
            remove_word_user(user_data[session_hash]["examples"][-1], session_hash)
        return update_user_fig(session_hash)
    return ""


def set_axis_user(axis_name, which_axis, from_words, to_words, session_hash):
    """
    Set a custom semantic axis for the 3D visualization.
    
    Args:
        axis_name (str): Name for the axis
        which_axis (str): Axis to modify (X, Y, or Z)
        from_words (str): Space or comma separated list of words for positive direction
        to_words (str): Space or comma separated list of words for negative direction
        session_hash (str): Unique identifier for the user session
        
    Returns:
        str: URL to the updated visualization
    """
    if axis_name != "residual":
        # Create a semantic axis from word pairs
        from_words, to_words = (
            from_words.replace(",", " ").split(),
            to_words.replace(",", " ").split(),
        )
        axis_emb = get_axis_embeddings(from_words, to_words)
        user_data[session_hash]["axis"][axisMap[which_axis]] = axis_emb
        user_data[session_hash]["axis_names"][axisMap[which_axis]] = axis_name

        # Update residual axis to be orthogonal to others
        for i, name in enumerate(user_data[session_hash]["axis_names"]):
            if name == "residual":
                user_data[session_hash]["axis"][i] = calculate_residual(
                    user_data[session_hash]["axis"],
                    user_data[session_hash]["axis_names"],
                    from_words,
                    to_words,
                    i,
                )
                user_data[session_hash]["axis_names"][i] = "residual"
    else:
        # Set axis to be residual (orthogonal to others)
        residual = calculate_residual(
            user_data[session_hash]["axis"],
            user_data[session_hash]["axis_names"],
            residual_axis=axisMap[which_axis],
        )
        user_data[session_hash]["axis"][axisMap[which_axis]] = residual
        user_data[session_hash]["axis_names"][axisMap[which_axis]] = axis_name

    # Recalculate coordinates with the new axes
    user_data[session_hash]["coords"] = (
        get_concat_embeddings(user_data[session_hash]["examples"])
        @ user_data[session_hash]["axis"].T
    )
    user_data[session_hash]["coords"][:, 1] = 5 * (
        1.0 - user_data[session_hash]["coords"][:, 1]
    )

    return update_user_fig(session_hash)


def generate_word_emb_vis(prompt, save_to_file=False, viz_dir=None):
    """
    Generate a visualization of a word's embedding vector.
    
    Args:
        prompt (str): Word or phrase to visualize
        save_to_file (bool): Whether to save the visualization to disk
        viz_dir (str): Directory to save the visualization in
        
    Returns:
        str: Base64-encoded image as data URL
    """
    # Get the word embedding and reshape for visualization
    emb = get_word_embeddings(prompt).reshape(77, 768)[1]

    # Create a heatmap visualization of the embedding
    plt.figure(figsize=(20, 3))
    plt.imshow([emb], cmap="inferno", aspect="auto")
    plt.xticks([])
    plt.yticks([])
    plt.title(f'{prompt}', pad=10, fontsize=48)
    plt.tight_layout()

    # Save to buffer
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()

    # Save to file if requested
    if save_to_file and viz_dir:
        os.makedirs(viz_dir, exist_ok=True)
        safe_filename = "".join([c if c.isalnum() else "_" for c in prompt])
        viz_path = os.path.join(viz_dir, f"{safe_filename}_emb.png")
        try:
            with open(viz_path, "wb") as f:
                f.write(buf.getvalue())
        except Exception as e:
            print(f"Error saving embedding to file: {e}")

    # Return as data URL
    img = "data:image/png;base64, " + base64.b64encode(buf.getvalue()).decode("utf-8")
    return img


def generate_word_embedding_visualization(word, session_hash):
    """
    Generate both the embedding visualization and corresponding image for a word.
    
    Args:
        word (str): Word to visualize
        session_hash (str): Unique identifier for the user session
        
    Returns:
        tuple: (
            Image: Embedding visualization,
            Image: Generated image for the word,
            str: Status message
        )
    """
    if not session_hash or not word:
        return None, None, "Invalid session or word"

    try:
        if session_hash not in user_data:
            return None, None, f"Invalid session"

        # Add word to visualization if not already present
        if word not in user_data[session_hash]["examples"]:
            add_rem_word_user(word, session_hash)

        # Get directories for storing outputs
        examples_dir = get_user_examples_dir(session_hash)
        viz_dir = get_user_viz_dir(session_hash)

        if not examples_dir or not viz_dir:
            return None, None, "Error: Could not create directories"

        # Generate embedding visualization
        str_viz_dir = str(viz_dir)
        emb_viz_b64 = generate_word_emb_vis(
            word, save_to_file=True, viz_dir=str_viz_dir
        )

        # Convert base64 to image
        emb_viz_bytes = base64.b64decode(emb_viz_b64.split(",")[1])
        emb_viz = Image.open(BytesIO(emb_viz_bytes))

        # Path for the generated image
        image_path = examples_dir / f"{get_safe_filename(word)}.jpg"

        # Load existing image or generate new one
        if image_path.exists():
            generated_img = Image.open(str(image_path))
        else:
            image = pipe(
                prompt=word,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            ).images[0]

            image.save(str(image_path), format="JPEG")
            generated_img = image

        return emb_viz, generated_img, f"Visualization for '{word}'"
    except Exception as e:
        print(f"Error generating visualization for '{word}': {e}")
        return None, None, f"Error: {str(e)}"


def load_user_gallery(session_hash):
    """
    Load the gallery of example images for a user session.
    
    Args:
        session_hash (str): Unique identifier for the user session
        
    Returns:
        list: List of (image, label) tuples for the gallery
    """
    if not session_hash:
        return []

    if session_hash not in user_data:
        return []

    example_images = []
    
    for example in user_data[session_hash]["examples"]:
        try:
            # Try to use cached image
            if example in user_data[session_hash]["images"]:
                img_str = user_data[session_hash]["images"][example]
                img = base64_to_image(img_str)
                if img:
                    example_images.append((img, example))
                    continue
            
            # Generate new image if needed
            image = pipe(
                prompt=example,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            ).images[0]
            
            # Cache the image
            if "images" not in user_data[session_hash]:
                user_data[session_hash]["images"] = {}
                
            user_data[session_hash]["images"][example] = image_to_base64(image)
            example_images.append((image, example))
            
        except Exception as e:
            print(f"Error handling image for gallery: '{example}': {e}")
            continue

    return example_images


def image_to_base64(img):
    """
    Convert a PIL image to a base64-encoded string.
    
    Args:
        img (PIL.Image): Image to convert
        
    Returns:
        str: Base64-encoded image data
    """
    if img is None:
        return ""
    img = img.resize((512, 512))  # Standardize size
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return img_str


def base64_to_image(img_str):
    """
    Convert a base64-encoded string to a PIL image.
    
    Args:
        img_str (str): Base64-encoded image data
        
    Returns:
        PIL.Image: Decoded image, or None if decoding fails
    """
    if not img_str:
        return None
    try:
        img_data = base64.b64decode(img_str)
        return Image.open(io.BytesIO(img_data))
    except:
        return None


def update_gallery_zip(session_hash, request=None):
    """
    Create a ZIP file containing all images in the user's gallery.
    
    Args:
        session_hash (str): Unique identifier for the user session
        request (gr.Request, optional): Gradio request containing session information
        
    Returns:
        str: Path to the created ZIP file, or None if creation failed
    """
    if not session_hash or session_hash not in user_data:
        return None
    
    # Collect all images from the session
    images_dict = user_data[session_hash]["images"]
    images_list = []
    for word, img_str in images_dict.items():
        if img_str is not None:
            img = base64_to_image(img_str)
            if img:
                images_list.append((img, word))
    
    if not images_list:
        return None
    
    # Create a ZIP file with all images
    from src.util import export_as_zip
    zip_path = export_as_zip(images_list, "embeddings", {}, request=request)
    return zip_path


# Export public functions from this module
__all__ = [
    "generate_user_html",
    "is_new_session",
    "init_user_session",
    "update_user_fig",
    "add_word_user",
    "remove_word_user",
    "add_rem_word_user",
    "change_word_user",
    "clear_words_user",
    "set_axis_user",
    "generate_word_emb_vis",
    "generate_word_embedding_visualization",
    "load_user_gallery",
    "update_gallery_zip",
]