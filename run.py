"""
Main Application Entry Point for Diffusion Demo

This module serves as the entry point for the Diffusion Demo application, orchestrating
the Gradio UI components, callback functions, and server initialization. It creates a
comprehensive interface for exploring various aspects of diffusion models, including:

1. Text-to-image generation with various parameters
2. Visualization of the denoising process
3. Exploration of latent space through perturbations, circular paths, and interpolation
4. CLIP embedding visualization and manipulation
5. Inpainting and negative prompt demonstrations

The application is structured as a multi-tab Gradio interface with detailed interactive
components and visualizations for educational purposes.
"""

import numpy as np
import gradio as gr
import os, io
from PIL import Image
from src.util import *
from src.pipelines import *
from serve import mount_aux_server
from src.util.session import session_manager
from src.pipelines.embeddings import user_data, update_gallery_zip

# Site visit tracking script
# js = """
# (function(window, document, dataLayerName, id) {
#     window[dataLayerName] = window[dataLayerName] || [];
#     window[dataLayerName].push({
#         start: (new Date).getTime(),
#         event: "stg.start"
#     });

#     var scripts = document.getElementsByTagName('script')[0];
#     var tags = document.createElement('script');

#     var qP = [];
#     if (dataLayerName !== "dataLayer") {
#         qP.push("data_layer_name=" + dataLayerName);
#     }
#     var qPString = qP.length > 0 ? ("?" + qP.join("&")) : "";

#     tags.async = true;
#     tags.src = "https://touretzky.containers.piwik.pro/" + id + ".js" + qPString;
#     scripts.parentNode.insertBefore(tags, scripts);

#     !function(a, n, i) {
#         a[n] = a[n] || {};
#         for (var c = 0; c < i.length; c++) {
#             !function(i) {
#                 a[n][i] = a[n][i] || {};
#                 a[n][i].api = a[n][i].api || function() {
#                     var a = [].slice.call(arguments, 0);
#                     if (typeof a[0] === "string") {
#                         window[dataLayerName].push({
#                             event: n + "." + i + ":" + a[0],
#                             parameters: [].slice.call(arguments, 1)
#                         });
#                     }
#                 }
#             }(i[c]);
#         }
#     }(window, "ppms", ["tm", "cm"]);
# })(window, document, 'dataLayer', '4b7bbce9-fa06-4d16-9dc6-6b0146eb8c31');
# """

# Initialize the main Gradio interface with a dark theme
with gr.Blocks(css="#step_size_circular {background-color: #FFFACD} #step_size_circular textarea {background-color: #FFFACD}", theme=gr.themes.Origin()) as demo:    # Main application header
    gr.Markdown("## Stable Diffusion Demo")
    
    # State variable to track the user's session across interactions
    session_hash_state = gr.State("")

    # ===== LATENT SPACE TAB =====
    # This tab contains demonstrations of various latent space manipulations
    with gr.Tab("Latent Space"):

        # ----- Beginner Section -----
        # Simple text-to-image generation for new users
        with gr.TabItem("Beginner"):
            gr.Markdown("Generate images from text.")

            with gr.Row():
                with gr.Column():
                    # Text prompt input
                    prompt_beginner = gr.Textbox(
                        lines=1,
                        label="Prompt",
                        value="close-up view of a kindly old man wearing a tartan cap, standing under a tree",
                    )

                    # Seed selection with visualization
                    with gr.Row(equal_height=True):
                        seed_beginner = gr.Slider(
                            minimum=0, maximum=100, step=1, value=14, label="Seed"
                        )
                        seed_vis_beginner = gr.Plot(
                            value=generate_seed_vis(14), label="Seed"
                        )

                    # Generation button
                    generate_images_button_beginner = gr.Button("Generate Image")

                # Output image display
                with gr.Column():
                    images_output_beginner = gr.Image(label="Image")

        # Update seed visualization when slider changes
        seed_beginner.change(
            fn=generate_seed_vis, inputs=[seed_beginner], outputs=[seed_vis_beginner]
        )

        # ----- Denoising Section -----
        # Visualize the step-by-step denoising process
        with gr.TabItem("Denoising"):
            gr.Markdown("Observe the intermediate images during denoising.")
            # Load HTML documentation for this feature
            gr.HTML(read_html("DiffusionDemo/html/denoising.html"))

            with gr.Row():
                with gr.Column():
                    # Text prompt input
                    prompt_denoise = gr.Textbox(
                        lines=1,
                        label="Prompt",
                        value="close-up view of a kindly old man wearing a tartan cap, standing under a tree",
                    )
                    
                    # Control for number of denoising steps
                    num_inference_steps_denoise = gr.Slider(
                        minimum=2,
                        maximum=100,
                        step=1,
                        value=8,
                        label="Number of Inference Steps",
                    )

                    # Seed selection with visualization
                    with gr.Row(equal_height=True):
                        seed_denoise = gr.Slider(
                            minimum=0, maximum=100, step=1, value=14, label="Seed"
                        )
                        seed_vis_denoise = gr.Plot(
                            value=generate_seed_vis(14), label="Seed"
                        )

                    # Generation button
                    generate_images_button_denoise = gr.Button("Generate Images")

                # Output displays
                with gr.Column():
                    # Gallery for multiple intermediate images
                    images_output_denoise = gr.Gallery(label="Images")
                    # Animated GIF of the denoising process
                    gif_denoise = gr.Image(label="GIF")
                    # Download option
                    zip_output_denoise = gr.File(label="Download ZIP")

        # Update seed visualization when slider changes
        seed_denoise.change(
            fn=generate_seed_vis, inputs=[seed_denoise], outputs=[seed_vis_denoise]
        )

        # ----- Seeds Section -----
        # Demonstrate how different random seeds affect image generation
        with gr.TabItem("Seeds"):
            gr.Markdown(
                "Understand how different starting points in latent space can lead to different images."
            )
            # Load HTML documentation for this feature
            gr.HTML(read_html("DiffusionDemo/html/seeds.html"))

            with gr.Row():
                with gr.Column():
                    # Text prompt input
                    prompt_seed = gr.Textbox(
                        lines=1,
                        label="Prompt",
                        value="close-up view of a kindly old man wearing a tartan cap, standing under a tree",
                    )
                    
                    # Starting seed parameter
                    starting_seed = gr.Slider(
                        minimum=0, maximum=100000, step=1, value=14, label="Starting Seed"
                    )
                    
                    # Control for number of different seeds to use
                    num_images_seed = gr.Slider(
                        minimum=1, maximum=100, step=1, value=5, label="Number of Seeds"
                    )
                    
                    # Control for inference steps per image
                    num_inference_steps_seed = gr.Slider(
                        minimum=2,
                        maximum=100,
                        step=1,
                        value=8,
                        label="Number of Inference Steps per Image",
                    )
                    
                    # Generation button
                    generate_images_button_seed = gr.Button("Generate Images")

                # Output displays
                with gr.Column():
                    # Gallery for multiple images from different seeds
                    images_output_seed = gr.Gallery(label="Images")
                    # Download option
                    zip_output_seed = gr.File(label="Download ZIP")

        # ----- Perturbations Section -----
        # Explore variations in latent space through controlled perturbations
        with gr.TabItem("Perturbations"):
            gr.Markdown("Explore different perturbations from a point in latent space.")
            # Load HTML documentation for this feature
            gr.HTML(read_html("DiffusionDemo/html/perturbations.html"))

            with gr.Row():
                with gr.Column():
                    # Text prompt input
                    prompt_perturb = gr.Textbox(
                        lines=1,
                        label="Prompt",
                        value="close-up view of a kindly old man wearing a tartan cap, standing under a tree",
                    )
                    
                    # Control for number of perturbed variations to create
                    num_images_perturb = gr.Slider(
                        minimum=0,
                        maximum=100,
                        step=1,
                        value=5,
                        label="Number of Perturbations",
                    )
                    
                    # Control for perturbation magnitude
                    perturbation_size_perturb = gr.Slider(
                        minimum=0,
                        maximum=1,
                        step=0.1,
                        value=0.1,
                        label="Perturbation Size",
                    )
                    
                    # Control for inference steps per image
                    num_inference_steps_perturb = gr.Slider(
                        minimum=2,
                        maximum=100,
                        step=1,
                        value=8,
                        label="Number of Inference Steps per Image",
                    )

                    # Seed selection with visualization
                    with gr.Row(equal_height=True):
                        seed_perturb = gr.Slider(
                            minimum=0, maximum=100, step=1, value=14, label="Seed"
                        )
                        seed_vis_perturb = gr.Plot(
                            value=generate_seed_vis(14), label="Seed"
                        )

                    # Generation button
                    generate_images_button_perturb = gr.Button("Generate Images")

                # Output displays
                with gr.Column():
                    # Gallery for perturbed images
                    images_output_perturb = gr.Gallery(label="Image")
                    # Download option
                    zip_output_perturb = gr.File(label="Download ZIP")

        # Update seed visualization when slider changes
        seed_perturb.change(
            fn=generate_seed_vis, inputs=[seed_perturb], outputs=[seed_vis_perturb]
        )

        # ----- Circular Section -----
        # Generate a circular path in latent space and observe how the images vary along the path
        with gr.TabItem("Circular"):
            gr.Markdown(
                "Generate a circular path in latent space and observe how the images vary along the path."
            )
            # Load HTML documentation for this feature
            gr.HTML(read_html("DiffusionDemo/html/circular.html"))

            with gr.Row():
                with gr.Column():
                    # Text prompt input
                    prompt_circular = gr.Textbox(
                        lines=1,
                        label="Prompt",
                        value="close-up view of a kindly old man wearing a tartan cap, standing under a tree",
                    )
                    # Control for number of steps around the circle
                    num_images_circular = gr.Slider(
                        minimum=1,
                        maximum=100,
                        step=1,
                        value=5,
                        label="Number of Steps around the Circle",
                    )

                    # Controls for start and end angles
                    with gr.Row():
                        start_degree_circular = gr.Slider(
                            minimum=0,
                            maximum=360,
                            step=1,
                            value=0,
                            label="Start Angle",
                            info="Enter the value in degrees",
                        )
                        end_degree_circular = gr.Slider(
                            minimum=0,
                            maximum=360,
                            step=1,
                            value=360,
                            label="End Angle",
                            info="Enter the value in degrees",
                        )
                        step_size_circular = gr.Textbox(
                            label="Step Size", value=360 / 5,
                            elem_id="step_size_circular"
                        )

                    # Control for number of inference steps per image
                    num_inference_steps_circular = gr.Slider(
                        minimum=2,
                        maximum=100,
                        step=1,
                        value=8,
                        label="Number of Inference Steps per Image",
                    )

                    # Seed selection with visualization
                    with gr.Row(equal_height=True):
                        seed_circular = gr.Slider(
                            minimum=0, maximum=100, step=1, value=14, label="Seed"
                        )
                        seed_vis_circular = gr.Plot(
                            value=generate_seed_vis(14), label="Seed"
                        )

                    # Generation button
                    generate_images_button_circular = gr.Button("Generate Images")

                with gr.Column():
                    # Gallery for multiple circular images
                    images_output_circular = gr.Gallery(label="Image")
                    # Animated GIF of the circular path
                    gif_circular = gr.Image(label="GIF")
                    # Download option
                    zip_output_circular = gr.File(label="Download ZIP")

        # Update step size when number of steps or angles change
        num_images_circular.change(
            fn=calculate_step_size,
            inputs=[num_images_circular, start_degree_circular, end_degree_circular],
            outputs=[step_size_circular],
        )
        start_degree_circular.change(
            fn=calculate_step_size,
            inputs=[num_images_circular, start_degree_circular, end_degree_circular],
            outputs=[step_size_circular],
        )
        end_degree_circular.change(
            fn=calculate_step_size,
            inputs=[num_images_circular, start_degree_circular, end_degree_circular],
            outputs=[step_size_circular],
        )
        # Update seed visualization when slider changes
        seed_circular.change(
            fn=generate_seed_vis, inputs=[seed_circular], outputs=[seed_vis_circular]
        )

        # ----- Poke Section -----
        # Perturb a region in the image and observe the effect
        with gr.TabItem("Poke"):
            gr.Markdown("Perturb a region in the image and observe the effect. Explores how changes to a small part of the latent image produce global effects in the target image. Changing only a single pixel has very subtle effects; changing larger regions has more dramatic effects.")
            # Load HTML documentation for this feature
            gr.HTML(read_html("DiffusionDemo/html/poke.html"))

            with gr.Row():
                with gr.Column():
                    # Text prompt input
                    prompt_poke = gr.Textbox(
                        lines=1,
                        label="Prompt",
                        value="close-up view of a kindly old man wearing a tartan cap, standing under a tree",
                    )
                    # Control for number of inference steps per image
                    num_inference_steps_poke = gr.Slider(
                        minimum=2,
                        maximum=100,
                        step=1,
                        value=8,
                        label="Number of Inference Steps per Image",
                    )

                    # Seed selection with visualization
                    with gr.Row(equal_height=True):
                        seed_poke = gr.Slider(
                            minimum=0, maximum=100, step=1, value=14, label="Seed"
                        )
                        seed_vis_poke = gr.Plot(
                            value=generate_seed_vis(14), label="Seed"
                        )

                    # Controls for poking region
                    pokeX = gr.Slider(
                        label="pokeX",
                        minimum=0,
                        maximum=64,
                        step=1,
                        value=32,
                        info="X coordinate of poke center",
                    )
                    pokeY = gr.Slider(
                        label="pokeY",
                        minimum=0,
                        maximum=64,
                        step=1,
                        value=32,
                        info="Y coordinate of poke center",
                    )
                    pokeHeight = gr.Slider(
                        label="pokeHeight",
                        minimum=1,
                        maximum=64,
                        step=1,
                        value=8,
                        info="Height of the poke",
                    )
                    pokeWidth = gr.Slider(
                        label="pokeWidth",
                        minimum=1,
                        maximum=64,
                        step=1,
                        value=8,
                        info="Width of the poke",
                    )

                    # Generation button
                    generate_images_button_poke = gr.Button("Generate Images")

                with gr.Column():
                    with gr.Row():
                        # Original images display (left two panels)
                        original_viz_output_poke = gr.Image(
                            value=visualize_poke(32, 32, 8, 8, 14)[0], label="Latent Image"
                        )

                        original_images_output_poke = gr.Image(
                            value=visualize_poke(32, 32, 8, 8, 14)[2], label="Original Image"
                        )

                    with gr.Row():                        
                        # Poked images display (right two panels)
                        poked_viz_output_poke = gr.Image(
                            value=visualize_poke(32, 32, 8, 8, 14)[1], label="Poked Latent Image"
                        )
                        poked_images_output_poke = gr.Image(
                            value=visualize_poke(32, 32, 8, 8, 14)[3], label="Poked Image"
                        )
                    # Download option
                    zip_output_poke = gr.File(label="Download ZIP")

        # Update poking visualization when parameters change
        pokeX.change(
            visualize_poke,
            inputs=[pokeX, pokeY, pokeHeight, pokeWidth, seed_poke],
            outputs=[original_viz_output_poke, poked_viz_output_poke, original_images_output_poke, poked_images_output_poke],
        )
        pokeY.change(
            visualize_poke,
            inputs=[pokeX, pokeY, pokeHeight, pokeWidth, seed_poke],
            outputs=[original_viz_output_poke, poked_viz_output_poke, original_images_output_poke, poked_images_output_poke],
        )
        pokeHeight.change(
            visualize_poke,
            inputs=[pokeX, pokeY, pokeHeight, pokeWidth, seed_poke],
            outputs=[original_viz_output_poke, poked_viz_output_poke, original_images_output_poke, poked_images_output_poke],
        )
        pokeWidth.change(
            visualize_poke,
            inputs=[pokeX, pokeY, pokeHeight, pokeWidth, seed_poke],
            outputs=[original_viz_output_poke, poked_viz_output_poke, original_images_output_poke, poked_images_output_poke],
        )
        # Update seed visualization when slider changes
        seed_poke.change(
            fn=generate_seed_vis, 
            inputs=[seed_poke],
            outputs=[seed_vis_poke],
        )

        seed_poke.change(
            fn=visualize_poke,
            inputs=[pokeX, pokeY, pokeHeight, pokeWidth, seed_poke],
            outputs=[original_viz_output_poke, poked_viz_output_poke, original_images_output_poke, poked_images_output_poke],
        )

        # ----- Guidance Section -----
        # Observe the effect of different guidance scales
        with gr.TabItem("Guidance"):
            gr.Markdown("Observe the effect of different guidance scales.")
            # Load HTML documentation for this feature
            gr.HTML(read_html("DiffusionDemo/html/guidance.html"))

            with gr.Row():
                with gr.Column():
                    # Text prompt input
                    prompt_guidance = gr.Textbox(
                        lines=1,
                        label="Prompt",
                        value="close-up view of a kindly old man wearing a tartan cap, standing under a tree",
                    )
                    # Control for number of inference steps per image
                    num_inference_steps_guidance = gr.Slider(
                        minimum=2,
                        maximum=100,
                        step=1,
                        value=8,
                        label="Number of Inference Steps per Image",
                    )
                    # Input for guidance scale values
                    guidance_scale_values = gr.Textbox(
                        lines=1, value="1, 8, 20, 30", label="Guidance Scale Values"
                    )

                    # Seed selection with visualization
                    with gr.Row(equal_height=True):
                        seed_guidance = gr.Slider(
                            minimum=0, maximum=100, step=1, value=14, label="Seed"
                        )
                        seed_vis_guidance = gr.Plot(
                            value=generate_seed_vis(14), label="Seed"
                        )

                    # Generation button
                    generate_images_button_guidance = gr.Button("Generate Images")

                with gr.Column():
                    # Gallery for multiple images with different guidance scales
                    images_output_guidance = gr.Gallery(label="Images")
                    # Download option
                    zip_output_guidance = gr.File(label="Download ZIP")

        # Update seed visualization when slider changes
        seed_guidance.change(
            fn=generate_seed_vis, inputs=[seed_guidance], outputs=[seed_vis_guidance]
        )

        # ----- Inpainting Section -----
        # Inpaint the image based on the prompt
        with gr.TabItem("Inpainting"):
            gr.Markdown("Inpaint the image based on the prompt.")
            # Load HTML documentation for this feature
            gr.HTML(read_html("DiffusionDemo/html/inpainting.html"))

            with gr.Row():
                with gr.Column():
                    # Uploaded image input
                    uploaded_img_inpaint = gr.Sketchpad(
                        sources="upload", brush=gr.Brush(colors=["#ffff00"], default_color="#ffff00"), type="pil", label="Upload"
                    )
                    # Text prompt input
                    prompt_inpaint = gr.Textbox(
                        lines=1, label="Prompt", value="sunglasses"
                    )
                    # Control for number of inference steps per image
                    num_inference_steps_inpaint = gr.Slider(
                        minimum=2,
                        maximum=100,
                        step=1,
                        value=8,
                        label="Number of Inference Steps per Image",
                    )

                    # Seed selection with visualization
                    with gr.Row(equal_height=True):
                        seed_inpaint = gr.Slider(
                            minimum=0, maximum=100, step=1, value=14, label="Seed"
                        )
                        seed_vis_inpaint = gr.Plot(
                            value=generate_seed_vis(14), label="Seed"
                        )

                    # Inpainting button
                    inpaint_button = gr.Button("Inpaint")

                with gr.Column():
                    # Inpainted image output
                    images_output_inpaint = gr.Image(label="Output")
                    # Download option
                    zip_output_inpaint = gr.File(label="Download ZIP")

        inpaint_button.click(
            fn=inpaint,
            inputs=[
                uploaded_img_inpaint,
                num_inference_steps_inpaint,
                seed_inpaint,
                prompt_inpaint,
            ],
            outputs=[images_output_inpaint, zip_output_inpaint],
        )
        seed_inpaint.change(
            fn=generate_seed_vis, inputs=[seed_inpaint], outputs=[seed_vis_inpaint]
        )

    # ===== CLIP SPACE TAB =====
    # This tab explores the CLIP text-image embedding space
    with gr.Tab("CLIP Space"):
        # ----- Embeddings Section -----
        # Visualize and explore text embeddings in 3D space
        with gr.TabItem("Embeddings"):
            gr.Markdown(
                "Visualize text embedding space in 3D with input texts and output images based on the chosen axis."
            )
            # Load HTML documentation for this feature
            gr.HTML(read_html("DiffusionDemo/html/embeddings.html"))

            # Container for the 3D visualization
            with gr.Row():
                output = gr.HTML(value="Loading...", elem_id="embedding-html")

            # Controls for adding/removing words and changing images
            with gr.Row(equal_height=True):
                word2add_rem = gr.Textbox(lines=1, label="Add/Remove word")
                word2change = gr.Textbox(lines=1, label="Change image for word")
                clear_words_button = gr.Button(value="Clear words")

            # Word embedding visualization section
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    # Input for visualizing specific word embeddings
                    word_input = gr.Textbox(
                        label="Visualize embedding for word", lines=1
                    )

                with gr.Column(scale=1):
                    # Display for embedding visualization
                    embedding_visualization = gr.Image(
                        type="pil", interactive=False, height="6vw"
                    )

            # Gallery of generated images for selected words
            with gr.Row():
                gallery = gr.Gallery(
                    label="Images of words",
                    show_label=True,
                    elem_id="gallery",
                    columns=4,
                    height="auto",
                    object_fit="contain",
                )
                
            # Download option for gallery
            with gr.Row(equal_height=True):
                zip_output_gallery = gr.File(label="Download Gallery ZIP", scale=1)

            # Advanced section for customizing semantic dimensions
            with gr.Accordion("Custom Semantic Dimensions", open=False):
                # First semantic dimension
                with gr.Row(equal_height=True):
                    axis_name_1 = gr.Textbox(label="Axis name", value="gender")
                    which_axis_1 = gr.Dropdown(
                        choices=["X - Axis", "Y - Axis", "Z - Axis", "---"],
                        value=whichAxisMap["which_axis_1"],
                        label="Axis direction",
                    )
                    from_words_1 = gr.Textbox(
                        lines=1,
                        label="Positive",
                        value="prince husband father son uncle",
                    )
                    to_words_1 = gr.Textbox(
                        lines=1,
                        label="Negative",
                        value="princess wife mother daughter aunt",
                    )
                    submit_1 = gr.Button("Submit")

                # Second semantic dimension
                with gr.Row(equal_height=True):
                    axis_name_2 = gr.Textbox(label="Axis name", value="age")
                    which_axis_2 = gr.Dropdown(
                        choices=["X - Axis", "Y - Axis", "Z - Axis", "---"],
                        value=whichAxisMap["which_axis_2"],
                        label="Axis direction",
                    )
                    from_words_2 = gr.Textbox(
                        lines=1, label="Positive", value="man woman king queen father"
                    )
                    to_words_2 = gr.Textbox(
                        lines=1, label="Negative", value="boy girl prince princess son"
                    )
                    submit_2 = gr.Button("Submit")

                # Third semantic dimension
                with gr.Row(equal_height=True):
                    axis_name_3 = gr.Textbox(label="Axis name", value="residual")
                    which_axis_3 = gr.Dropdown(
                        choices=["X - Axis", "Y - Axis", "Z - Axis", "---"],
                        value=whichAxisMap["which_axis_3"],
                        label="Axis direction",
                    )
                    from_words_3 = gr.Textbox(lines=1, label="Positive")
                    to_words_3 = gr.Textbox(lines=1, label="Negative")
                    submit_3 = gr.Button("Submit")

                # Fourth semantic dimension
                with gr.Row(equal_height=True):
                    axis_name_4 = gr.Textbox(label="Axis name", value="number")
                    which_axis_4 = gr.Dropdown(
                        choices=["X - Axis", "Y - Axis", "Z - Axis", "---"],
                        value=whichAxisMap["which_axis_4"],
                        label="Axis direction",
                    )
                    from_words_4 = gr.Textbox(
                        lines=1,
                        label="Positive",
                        value="boys girls cats puppies computers",
                    )
                    to_words_4 = gr.Textbox(
                        lines=1, label="Negative", value="boy girl cat puppy computer"
                    )
                    submit_4 = gr.Button("Submit")

                # Fifth semantic dimension
                with gr.Row(equal_height=True):
                    axis_name_5 = gr.Textbox(label="Axis name", value="royalty")
                    which_axis_5 = gr.Dropdown(
                        choices=["X - Axis", "Y - Axis", "Z - Axis", "---"],
                        value=whichAxisMap["which_axis_5"],
                        label="Axis direction",
                    )
                    from_words_5 = gr.Textbox(
                        lines=1,
                        label="Positive",
                        value="king queen prince princess duchess",
                    )
                    to_words_5 = gr.Textbox(
                        lines=1, label="Negative", value="man woman boy girl woman"
                    )
                    submit_5 = gr.Button("Submit")

                # Sixth semantic dimension (empty by default)
                with gr.Row(equal_height=True):
                    axis_name_6 = gr.Textbox(label="Axis name")
                    which_axis_6 = gr.Dropdown(
                        choices=["X - Axis", "Y - Axis", "Z - Axis", "---"],
                        value=whichAxisMap["which_axis_6"],
                        label="Axis direction",
                    )
                    from_words_6 = gr.Textbox(lines=1, label="Positive")
                    to_words_6 = gr.Textbox(lines=1, label="Negative")
                    submit_6 = gr.Button("Submit")

            # Storage for embeddings data across sessions
            embeddings_storage = gr.BrowserState()

        @demo.load(inputs=[embeddings_storage, session_hash_state], outputs=[embeddings_storage, zip_output_gallery])
        def init_storage(storage, session_hash):
            """Initialize browser storage and restore previous session data if available"""
            if not session_hash:
                return storage, None
            
            if storage is None:
                storage = {}
            
            # If this session has previous examples, restore them
            if session_hash in storage and "examples" in storage[session_hash]:
                if session_hash not in user_data:
                    # Initialize user data with defaults
                    user_data[session_hash] = {
                        "examples": [],
                        "images": {},
                        "coords": np.array([]),
                        "axis": axis.copy(),
                        "axis_names": axis_names.copy(),
                    }
                
                # Restore previous examples and images
                user_data[session_hash]["examples"] = storage[session_hash]["examples"].copy()
                if "images" in storage[session_hash]:
                    user_data[session_hash]["images"] = storage[session_hash]["images"].copy()
            
            # Update the gallery ZIP file
            zip_path = update_gallery_zip(session_hash)
            return storage, zip_path

        def load_user_html(request: gr.Request):
            """Initialize user session and load the 3D visualization"""
            # Get or create a session for this user
            flask_url, session_hash, is_new = init_user_session(request)
            
            # Create an iframe to display the interactive 3D plot
            html_content = f"""
            <iframe id="html-frame" src="{flask_url}" style="width:100%; height:700px;"></iframe>
            """
            
            # Notify if this is a new session
            if is_new:
                gr.Info("New session initialized.")
                embeddings_storage.value = {
                    "images": {},
                    "embeddings": {},
                    "visualizations": {}
                }

            # Load gallery images and update ZIP file
            gallery_images = load_user_gallery(session_hash)
            zip_path = update_gallery_zip(session_hash, request)
            return html_content, session_hash, gallery_images, zip_path

        # Connect the load function to the demo startup
        demo.load(load_user_html, None, [output, session_hash_state, gallery, zip_output_gallery])

        # Handler for adding or removing words
        @word2add_rem.submit(
            inputs=[word2add_rem, session_hash_state, embeddings_storage],
            outputs=[output, word2add_rem, gallery, embeddings_storage, zip_output_gallery]
        )
        def add_rem_word_handler(words, session_hash, storage):
            """Add or remove words from the 3D visualization"""
            # Initialize storage if needed
            if storage is None:
                storage = {
                    "images": {},
                    "embeddings": {},
                    "visualizations": {}
                }
                
            # Call backend function to add/remove words and get updated visualization URL
            flask_url = add_rem_word_user(words, session_hash)
            
            # Create iframe HTML to display the updated visualization
            html_content = f"""
            <iframe id="html-frame" src="{flask_url}" style="width:100%; height:700px;"></iframe>
            """
            
            # Update gallery with current images
            gallery_images = load_user_gallery(session_hash)
            
            # Initialize session storage if needed
            if session_hash not in storage:
                storage[session_hash] = {"examples": [], "images": {}}
            
            # Update storage with current user data
            if session_hash in user_data:
                storage[session_hash]["examples"] = user_data[session_hash]["examples"].copy()
                if "images" in user_data[session_hash]:
                    storage[session_hash]["images"] = user_data[session_hash]["images"].copy()
            
            # Update the gallery ZIP file
            zip_path = update_gallery_zip(session_hash)
            return html_content, "", gallery_images, storage, zip_path

        # Handler for changing word images
        @word2change.submit(
            inputs=[word2change, session_hash_state, embeddings_storage],
            outputs=[output, word2change, gallery, embeddings_storage, zip_output_gallery]
        )
        def change_word_handler(word, session_hash, storage):
            """Generate a new image for an existing word in the visualization"""
            # Initialize storage if needed
            if storage is None:
                storage = {
                    "images": {},
                    "embeddings": {},
                    "visualizations": {}
                }
                
            # Call backend function to change the word's image and get updated visualization URL
            flask_url = change_word_user(word, session_hash)
            
            # Create iframe HTML to display the updated visualization
            html_content = f"""
            <iframe id="html-frame" src="{flask_url}" style="width:100%; height:700px;"></iframe>
            """
            
            # Update gallery with current images
            gallery_images = load_user_gallery(session_hash)
            
            # Initialize session storage if needed
            if session_hash not in storage:
                storage[session_hash] = {"examples": [], "images": {}}
            
            # Update storage with current user data
            if session_hash in user_data:
                storage[session_hash]["examples"] = user_data[session_hash]["examples"].copy()
                if "images" in user_data[session_hash]:
                    storage[session_hash]["images"] = user_data[session_hash]["images"].copy()
            
            # Update the gallery ZIP file
            zip_path = update_gallery_zip(session_hash)
            return html_content, "", gallery_images, storage, zip_path

        # Handler for clearing all words
        @clear_words_button.click(
            inputs=[session_hash_state, embeddings_storage],
            outputs=[output, gallery, embeddings_storage, zip_output_gallery]
        )
        def clear_words_handler(session_hash, storage):
            """Clear all words from the 3D visualization"""
            # Initialize storage if needed
            if storage is None:
                storage = {
                    "images": {},
                    "embeddings": {},
                    "visualizations": {}
                }
                
            # Call backend function to clear words and get updated visualization URL
            clear_url = clear_words_user(session_hash)
            
            # Create iframe HTML to display the updated visualization
            html_content = f"""
            <iframe id="html-frame" src="{clear_url}" style="width:100%; height:700px;"></iframe>
            """
            
            # Update gallery with current images (should be empty after clearing)
            gallery_images = load_user_gallery(session_hash)
            
            # Reset session storage for this user
            if session_hash in storage:
                storage[session_hash] = {"examples": [], "images": {}}
            return html_content, gallery_images, storage, None

        @submit_1.click(
            inputs=[
                axis_name_1,
                which_axis_1,
                from_words_1,
                to_words_1,
                session_hash_state,
            ],
            outputs=[
                output, 
                gallery,
                which_axis_2,
                which_axis_3,
                which_axis_4,
                which_axis_5,
                which_axis_6,
            ],
        )
        def set_axis_wrapper(axis_name, which_axis, from_words, to_words, session_hash):
            """Set the first semantic dimension and update the visualization"""
            # Ensure no duplicate axis assignments by clearing this axis assignment from other dimensions
            for ax in whichAxisMap:
                if whichAxisMap[ax] == which_axis:
                    whichAxisMap[ax] = "---"
                    
            # Assign the axis to this dimension
            whichAxisMap["which_axis_1"] = which_axis

            # Call backend function to set the axis and get updated visualization URL
            flask_url = set_axis_user(
                axis_name, which_axis, from_words, to_words, session_hash
            )
            
            # Create iframe HTML to display the updated visualization
            html_content = f"""
            <iframe id="html-frame" src="{flask_url}" style="width:100%; height:700px;"></iframe>
            """
            
            # Update gallery with current images
            gallery_images = load_user_gallery(session_hash)
            
            # Return updated visualization and axis dropdowns
            return (
                html_content, 
                gallery_images,
                whichAxisMap["which_axis_2"],
                whichAxisMap["which_axis_3"],
                whichAxisMap["which_axis_4"],
                whichAxisMap["which_axis_5"],
                whichAxisMap["which_axis_6"],
            )
        
        @submit_2.click(
            inputs=[
                axis_name_2,
                which_axis_2,
                from_words_2,
                to_words_2,
                session_hash_state,
            ],
            outputs=[
                output, 
                gallery,
                which_axis_1,
                which_axis_3,
                which_axis_4,
                which_axis_5,
                which_axis_6,
            ],
        )
        def set_axis_wrapper(axis_name, which_axis, from_words, to_words, session_hash):
            """Set the second semantic dimension and update the visualization"""
            # Ensure no duplicate axis assignments
            for ax in whichAxisMap:
                if whichAxisMap[ax] == which_axis:
                    whichAxisMap[ax] = "---"
                    
            whichAxisMap["which_axis_2"] = which_axis

            flask_url = set_axis_user(
                axis_name, which_axis, from_words, to_words, session_hash
            )
            html_content = f"""
            <iframe id="html-frame" src="{flask_url}" style="width:100%; height:700px;"></iframe>
            """
            gallery_images = load_user_gallery(session_hash)
            return (
                html_content, 
                gallery_images,
                whichAxisMap["which_axis_1"],
                whichAxisMap["which_axis_3"],
                whichAxisMap["which_axis_4"],
                whichAxisMap["which_axis_5"],
                whichAxisMap["which_axis_6"],
            )
        
        @submit_3.click(
            inputs=[
                axis_name_3,
                which_axis_3,
                from_words_3,
                to_words_3,
                session_hash_state,
            ],
            outputs=[
                output, 
                gallery,
                which_axis_1,
                which_axis_2,
                which_axis_4,
                which_axis_5,
                which_axis_6,
            ],
        )
        def set_axis_wrapper(axis_name, which_axis, from_words, to_words, session_hash):
            """Set the third semantic dimension and update the visualization"""
            # Ensure no duplicate axis assignments
            for ax in whichAxisMap:
                if whichAxisMap[ax] == which_axis:
                    whichAxisMap[ax] = "---"
                    
            whichAxisMap["which_axis_3"] = which_axis

            flask_url = set_axis_user(
                axis_name, which_axis, from_words, to_words, session_hash
            )
            html_content = f"""
            <iframe id="html-frame" src="{flask_url}" style="width:100%; height:700px;"></iframe>
            """
            gallery_images = load_user_gallery(session_hash)
            return (
                html_content, 
                gallery_images,
                whichAxisMap["which_axis_1"],
                whichAxisMap["which_axis_2"],
                whichAxisMap["which_axis_4"],
                whichAxisMap["which_axis_5"],
                whichAxisMap["which_axis_6"],
            )
        
        @submit_4.click(
            inputs=[
                axis_name_4,
                which_axis_4,
                from_words_4,
                to_words_4,
                session_hash_state,
            ],
            outputs=[
                output, 
                gallery,
                which_axis_1,
                which_axis_2,
                which_axis_3,
                which_axis_5,
                which_axis_6,
            ],
        )
        def set_axis_wrapper(axis_name, which_axis, from_words, to_words, session_hash):
            """Set the fourth semantic dimension and update the visualization"""
            # Ensure no duplicate axis assignments
            for ax in whichAxisMap:
                if whichAxisMap[ax] == which_axis:
                    whichAxisMap[ax] = "---"
                    
            whichAxisMap["which_axis_4"] = which_axis

            flask_url = set_axis_user(
                axis_name, which_axis, from_words, to_words, session_hash
            )
            html_content = f"""
            <iframe id="html-frame" src="{flask_url}" style="width:100%; height:700px;"></iframe>
            """
            gallery_images = load_user_gallery(session_hash)
            return (
                html_content, 
                gallery_images,
                whichAxisMap["which_axis_1"],
                whichAxisMap["which_axis_2"],
                whichAxisMap["which_axis_3"],
                whichAxisMap["which_axis_5"],
                whichAxisMap["which_axis_6"],
            )
        
        @submit_5.click(
            inputs=[
                axis_name_5,
                which_axis_5,
                from_words_5,
                to_words_5,
                session_hash_state,
            ],
            outputs=[
                output, 
                gallery,
                which_axis_1,
                which_axis_2,
                which_axis_3,
                which_axis_4,
                which_axis_6,
            ],
        )
        def set_axis_wrapper(axis_name, which_axis, from_words, to_words, session_hash):
            """Set the fifth semantic dimension and update the visualization"""
            # Ensure no duplicate axis assignments
            for ax in whichAxisMap:
                if whichAxisMap[ax] == which_axis:
                    whichAxisMap[ax] = "---"
                    
            whichAxisMap["which_axis_5"] = which_axis

            flask_url = set_axis_user(
                axis_name, which_axis, from_words, to_words, session_hash
            )
            html_content = f"""
            <iframe id="html-frame" src="{flask_url}" style="width:100%; height:700px;"></iframe>
            """
            gallery_images = load_user_gallery(session_hash)
            return (
                html_content, 
                gallery_images,
                whichAxisMap["which_axis_1"],
                whichAxisMap["which_axis_2"],
                whichAxisMap["which_axis_3"],
                whichAxisMap["which_axis_4"],
                whichAxisMap["which_axis_6"],
            )
        
        @submit_6.click(
            inputs=[
                axis_name_6,
                which_axis_6,
                from_words_6,
                to_words_6,
                session_hash_state,
            ],
            outputs=[
                output, 
                gallery,
                which_axis_1,
                which_axis_2,
                which_axis_3,
                which_axis_4,
                which_axis_5,
            ],
        )
        def set_axis_wrapper(axis_name, which_axis, from_words, to_words, session_hash):
            """Set the sixth semantic dimension and update the visualization"""
            # Ensure no duplicate axis assignments
            for ax in whichAxisMap:
                if whichAxisMap[ax] == which_axis:
                    whichAxisMap[ax] = "---"
                    
            whichAxisMap["which_axis_6"] = which_axis

            flask_url = set_axis_user(
                axis_name, which_axis, from_words, to_words, session_hash
            )
            html_content = f"""
            <iframe id="html-frame" src="{flask_url}" style="width:100%; height:700px;"></iframe>
            """
            gallery_images = load_user_gallery(session_hash)
            return (
                html_content, 
                gallery_images,
                whichAxisMap["which_axis_1"],
                whichAxisMap["which_axis_2"],
                whichAxisMap["which_axis_3"],
                whichAxisMap["which_axis_4"],
                whichAxisMap["which_axis_5"],
            )

        @word_input.submit(
            inputs=[word_input, session_hash_state],
            outputs=[embedding_visualization, word_input, gallery, output, zip_output_gallery],
        )
        def handle_word_visualization(word, session_hash):
            """Generate and display the embedding visualization for a specific word"""
            # Skip if the input is empty
            if not word.strip():
                return None, "", load_user_gallery(session_hash), output.value, None

            # Generate the embedding visualization and image for this word
            emb_viz, generated_img, label = generate_word_embedding_visualization(
                word, session_hash
            )

            # Update the 3D visualization
            flask_url = update_user_fig(session_hash)
            html_content = f"""
            <iframe id="html-frame" src="{flask_url}" style="width:100%; height:700px;"></iframe>
            """
            
            # Update the gallery ZIP file
            zip_path = update_gallery_zip(session_hash)
            return emb_viz, "", load_user_gallery(session_hash), html_content, zip_path

        # ----- Interpolate Section -----
        # Demonstrate interpolation between two text prompts
        with gr.TabItem("Interpolate"):
            gr.Markdown(
                "Interpolate between the first and the second prompt, and observe how the output changes."
            )
            # Load HTML documentation for this feature
            gr.HTML(read_html("DiffusionDemo/html/interpolate.html"))

            with gr.Row():
                with gr.Column():
                    # Input for the starting prompt
                    promptA = gr.Textbox(
                        lines=1,
                        label="First Prompt",
                        value="close-up view of a kindly old man wearing a tartan cap, standing under a tree",
                    )
                    # Input for the ending prompt
                    promptB = gr.Textbox(
                        lines=1,
                        label="Second Prompt",
                        value="close-up view of a kindly old woman wearing a tartan cap, standing under a tree",
                    )
                    # Control for number of intermediate steps
                    num_images_interpolate = gr.Slider(
                        minimum=0,
                        maximum=100,
                        step=1,
                        value=5,
                        label="Number of Interpolation Steps",
                    )
                    # Control for inference steps per image
                    num_inference_steps_interpolate = gr.Slider(
                        minimum=2,
                        maximum=100,
                        step=1,
                        value=8,
                        label="Number of Inference Steps per Image",
                    )

                    # Seed selection with visualization
                    with gr.Row(equal_height=True):
                        seed_interpolate = gr.Slider(
                            minimum=0, maximum=100, step=1, value=14, label="Seed"
                        )
                        seed_vis_interpolate = gr.Plot(
                            value=generate_seed_vis(14), label="Seed"
                        )

                    # Generation button
                    generate_images_button_interpolate = gr.Button("Generate Images")

                # Output displays
                with gr.Column():
                    # Gallery for interpolated images
                    images_output_interpolate = gr.Gallery(label="Interpolated Images")
                    # Animated GIF of the interpolation sequence
                    gif_interpolate = gr.Image(label="GIF")
                    # Download option
                    zip_output_interpolate = gr.File(label="Download ZIP")

        # Update seed visualization when slider changes
        seed_interpolate.change(
            fn=generate_seed_vis,
            inputs=[seed_interpolate],
            outputs=[seed_vis_interpolate],
        )

        # ----- Negative Section -----
        # Demonstrate the effect of negative prompts on image generation
        with gr.TabItem("Negative"):
            gr.Markdown("Observe the effect of negative prompts.")
            # Load HTML documentation for this feature
            gr.HTML(read_html("DiffusionDemo/html/negative.html"))

            with gr.Row():
                with gr.Column():
                    # Input for positive prompt
                    prompt_negative = gr.Textbox(
                        lines=1,
                        label="Prompt",
                        value="close-up view of a kindly old man wearing a tartan cap, standing under a tree",
                    )
                    # Input for negative prompt (what to avoid)
                    neg_prompt = gr.Textbox(
                        lines=1, label="Negative Prompt", value="Yellow"
                    )
                    # Control for inference steps
                    num_inference_steps_negative = gr.Slider(
                        minimum=2,
                        maximum=100,
                        step=1,
                        value=8,
                        label="Number of Inference Steps per Image",
                    )

                    # Seed selection with visualization
                    with gr.Row(equal_height=True):
                        seed_negative = gr.Slider(
                            minimum=0, maximum=100, step=1, value=14, label="Seed"
                        )
                        seed_vis_negative = gr.Plot(
                            value=generate_seed_vis(14), label="Seed"
                        )

                    # Generation button
                    generate_images_button_negative = gr.Button("Generate Images")

                # Output displays
                with gr.Column():
                    # Image without negative prompt applied
                    images_output_negative = gr.Image(
                        label="Image without Negative Prompt"
                    )
                    # Image with negative prompt applied
                    images_neg_output_negative = gr.Image(
                        label="Image with Negative Prompt"
                    )
                    # Download option
                    zip_output_negative = gr.File(label="Download ZIP")

        # ----- Filter Section -----
        # Demonstrate the effect of content filtering
        with gr.TabItem("Filter"):
            gr.Markdown("Demonstrate how content filtering works with different thresholds.")
            # Load HTML documentation for this feature
            gr.HTML(read_html("DiffusionDemo/html/filter.html"))

            with gr.Row():
                with gr.Column():
                    # Text prompt input
                    prompt_filter = gr.Textbox(
                        lines=1,
                        label="Prompt",
                        value="A child with their pet",
                    )
                    # Filter concept input
                    filter_concept = gr.Textbox(
                        lines=1,
                        label="Filter Concept",
                        value="cat",
                    )
                    # Control for number of images to generate
                    num_images_filter = gr.Slider(
                        minimum=1, maximum=20, step=1, value=5, label="Number of Images"
                    )
                    # Control for inference steps per image
                    num_inference_steps_filter = gr.Slider(
                        minimum=2,
                        maximum=100,
                        step=1,
                        value=8,
                        label="Number of Inference Steps per Image",
                    )
                    # Control for filter threshold
                    filter_threshold = gr.Slider(
                        minimum=-1,
                        maximum=1,
                        step=0.001,
                        value=0.03,
                        label="Filter Threshold",
                    )
                    
                    # Generation button
                    generate_images_button_filter = gr.Button("Generate Images")

                # Output displays
                with gr.Column():
                    # Gallery for filtered images
                    images_output_filter = gr.Gallery(label="Images with Filter Scores")
                    # Download option
                    zip_output_filter = gr.File(label="Download ZIP")

        # Connect Filter tab button to backend function
        generate_images_button_filter.click(
            fn=display_filter_images,
            inputs=[
                prompt_filter,
                filter_concept,
                num_images_filter,
                num_inference_steps_filter,
                filter_threshold,
            ],
            outputs=[images_output_filter, zip_output_filter],
        )

        # Update seed visualization when slider changes
        seed_negative.change(
            fn=generate_seed_vis, inputs=[seed_negative], outputs=[seed_vis_negative]
        )

        # ----- Event Handlers for Latent Space Tabs -----
        # These functions connect UI buttons to backend processing

        # Handler for the Beginner tab's generate button
        @generate_images_button_beginner.click(
            inputs=[prompt_beginner, seed_beginner],
            outputs=[images_output_beginner, prompt_denoise, prompt_seed, prompt_perturb, prompt_circular, prompt_poke, prompt_guidance, prompt_negative, promptA],
        )
        def generate_images_wrapper(
            prompt, seed, progress=gr.Progress(), request: gr.Request = None
        ):
            """Generate a single image in the Beginner tab and propagate the prompt to other tabs"""
            # Use the poke function with poke=False to just generate a regular image
            images, _ = display_poke_images(
                prompt, seed, num_inference_steps=8, poke=False, intermediate=False, request=request
            )
            # Return the image and propagate the prompt to all other tabs
            return [images] + [prompt] * 8

        # Handler for the Denoising tab's generate button
        @generate_images_button_denoise.click(
            inputs=[prompt_denoise, seed_denoise, num_inference_steps_denoise],
            outputs=[images_output_denoise, gif_denoise, zip_output_denoise, promptA, prompt_beginner, prompt_seed, prompt_perturb, prompt_circular, prompt_poke, prompt_guidance, prompt_negative],
        )
        def generate_images_wrapper(
            prompt, seed, num_inference_steps, progress=gr.Progress(), request: gr.Request = None
        ):
            """Generate intermediate denoising steps for visualization and create a GIF animation"""
            # Generate images with intermediate steps enabled
            images, _ = display_poke_images(
                prompt, seed, num_inference_steps, poke=False, intermediate=True, request=request
            )
            
            # Create metadata for the exports
            fname = "denoising"
            tab_config = {
                "Tab": "Denoising",
                "Prompt": prompt,
                "Number of Inference Steps": num_inference_steps,
                "Seed": seed,
            }
            
            # Create ZIP file for download
            zip_path = export_as_zip(images, fname, tab_config, request=request)
            
            # Create animated GIF from the intermediate images
            progress(1, desc="Exporting as gif")
            gif_path = export_as_gif(images, filename="denoising.gif", request=request)
            
            # Return outputs and propagate prompt to other tabs
            return [gr.Gallery(label="Images", value=images), gif_path, zip_path] + [prompt] * 8

        # Connect Seeds tab button to backend function
        generate_images_button_seed.click(
            fn=display_seed_images,
            inputs=[prompt_seed, num_inference_steps_seed, num_images_seed, starting_seed], 
            outputs=[images_output_seed, zip_output_seed, promptA, prompt_beginner, prompt_denoise, prompt_perturb, prompt_circular, prompt_poke, prompt_guidance, prompt_negative],
        )

        # Connect Perturbations tab button to backend function
        generate_images_button_perturb.click(
            fn=display_perturb_images,
            inputs=[
                prompt_perturb,
                seed_perturb,
                num_inference_steps_perturb,
                num_images_perturb,
                perturbation_size_perturb,
            ],
            outputs=[images_output_perturb, zip_output_perturb, promptA, prompt_beginner, prompt_denoise, prompt_seed, prompt_circular, prompt_poke, prompt_guidance, prompt_negative],
        )

        # Connect Circular tab button to backend function
        generate_images_button_circular.click(
            fn=display_circular_images,
            inputs=[
                prompt_circular,
                seed_circular,
                num_inference_steps_circular,
                num_images_circular,
                start_degree_circular,
                end_degree_circular,
            ],
            outputs=[images_output_circular, gif_circular, zip_output_circular, promptA, prompt_beginner, prompt_denoise, prompt_seed, prompt_perturb, prompt_poke, prompt_guidance, prompt_negative],
        )

        # Handler for the Poke tab's generate button
        @generate_images_button_poke.click(
            inputs=[
                prompt_poke,
                seed_poke,
                num_inference_steps_poke,
                pokeX,
                pokeY,
                pokeHeight,
                pokeWidth,
            ],
            outputs=[
                original_viz_output_poke,
                poked_viz_output_poke,
                original_images_output_poke,
                poked_images_output_poke,
                zip_output_poke,
                promptA,
                prompt_beginner,
                prompt_denoise,
                prompt_seed,
                prompt_perturb,
                prompt_circular,
                prompt_guidance,
                prompt_negative,
            ],
        )
        def generate_images_wrapper(
            prompt,
            seed,
            num_inference_steps,
            pokeX=pokeX,
            pokeY=pokeY,
            pokeHeight=pokeHeight,
            pokeWidth=pokeWidth,
            request: gr.Request = None
        ):
            """Generate both original and poked images with visualization of the modified region"""
            # Generate the actual images with the poke operation
            _, _ = display_poke_images(
                prompt,
                seed,
                num_inference_steps,
                poke=True,
                pokeX=pokeX,
                pokeY=pokeY,
                pokeHeight=pokeHeight,
                pokeWidth=pokeWidth,
                intermediate=False,
                request=request
            )
            
            # Visualize the poke region on both images - using the user's seed
            blank1, blank2, images, modImages = visualize_poke(
                pokeX, pokeY, pokeHeight, pokeWidth, seed=seed, request=request
            )
            
            # Create metadata for the exports
            fname = "poke"
            tab_config = {
                "Tab": "Poke",
                "Prompt": prompt,
                "Number of Inference Steps per Image": num_inference_steps,
                "Seed": seed,
                "PokeX": pokeX,
                "PokeY": pokeY,
                "PokeHeight": pokeHeight,
                "PokeWidth": pokeWidth,
            }
            
            # Prepare images for ZIP export
            imgs_list = []
            imgs_list.append((blank1, "Visualization Original"))
            imgs_list.append((blank2, "Visualization Poked"))
            imgs_list.append((images, "Original Image"))
            imgs_list.append((modImages, "Poked Image"))
            
            # Create ZIP file for download
            zip_path = export_as_zip(imgs_list, fname, tab_config, request=request)
            
            # Return outputs and propagate prompt to other tabs
            return [blank1, blank2, images, modImages, zip_path] + [prompt] * 8
        
        # Connect Guidance tab button to backend function
        generate_images_button_guidance.click(
            fn=display_guidance_images,
            inputs=[
                prompt_guidance,
                seed_guidance,
                num_inference_steps_guidance,
                guidance_scale_values,
            ],
            outputs=[images_output_guidance, zip_output_guidance, promptA, prompt_beginner, prompt_denoise, prompt_seed, prompt_perturb, prompt_circular, prompt_poke, prompt_negative],
        )

        # Connect Interpolate tab button to backend function
        generate_images_button_interpolate.click(
            fn=display_interpolate_images,
            inputs=[
                seed_interpolate,
                promptA,
                promptB,
                num_inference_steps_interpolate,
                num_images_interpolate,
            ],
            outputs=[
                images_output_interpolate,
                gif_interpolate,
                zip_output_interpolate,
                prompt_beginner,
                prompt_denoise,
                prompt_seed,
                prompt_perturb,
                prompt_circular,
                prompt_poke,
                prompt_guidance,
                prompt_negative,
            ],
        )

        # Connect Negative tab button to backend function
        generate_images_button_negative.click(
            fn=display_negative_images,
            inputs=[
                prompt_negative,
                seed_negative,
                num_inference_steps_negative,
                neg_prompt,
            ],
            outputs=[
                images_output_negative,
                images_neg_output_negative,
                zip_output_negative,
                promptA,
                prompt_beginner,
                prompt_denoise,
                prompt_seed,
                prompt_perturb,
                prompt_circular,
                prompt_poke,
                prompt_guidance,
            ],
        )

    # ===== ABOUT TAB =====
    # Provides information about the demo, model, and credits
    with gr.Tab("About"):
        gr.Markdown("""
        This demonstration provides an in-depth exploration of stable diffusion models, walking you through the inner workings in a step-by-step manner. You can experiment with various aspects of the model to better understand how text-to-image generation works.
        
        ### Diffusion Model Used
        
        This demo uses [Dreamshaper 8](https://huggingface.co/lykon/dreamshaper-8), a Stable Diffusion model fine-tuned on stable-diffusion-v1-5. Dreamshaper 8 is designed to handle both artistic and realistic styles, with good capabilities for both photorealism and anime-style content.
        
        ### NSFW Content Filtering
        
        This demo implements content safety using the Stable Diffusion Safety Checker component. The safety system:
        
        - Uses a CLIP-based model to analyze generated images
        - Compares image embeddings to known unsafe content patterns
        - Blocks generation of potentially inappropriate imagery
        
        ### References
        
        If you're interested in learning more about diffusion models:
        
        - [Cats in Latent Space](https://youtu.be/hb-KT66rCT8?si=RyCQCG7qU-iBugjV) - An intuitive visual explanation of latent diffusion
        - [Diffusion Explainer: Stable Diffusion Explained with Visualization](https://poloclub.github.io/diffusion-explainer/) - Interactive explanation of diffusion models
        - [The Illustrated Stable Diffusion](https://jalammar.github.io/illustrated-stable-diffusion/) - Visual guide to understanding Stable Diffusion
        - [Red-Teaming the Stable Diffusion Safety Filter](https://arxiv.org/abs/2210.04610) - Research paper explaining Stable Diffusion safety filter
        
        ### Credits
        
        Author: Adithya Kameswara Rao, Carnegie Mellon University.
                    
        Advisor: David S. Touretzky, Carnegie Mellon University.
                    
        This work was funded by a grant from NEOM Company, and by National Science Foundation award IIS-2112633.
        """)
    

# ===== MAIN APPLICATION EXECUTION =====
# Run the application when executed as a script
if __name__ == "__main__":
    # Start the session cleanup thread to remove old sessions
    session_manager.start_cleanup_thread()

    try:
        # Create outputs directory if it doesn't exist
        os.makedirs("outputs", exist_ok=True)

        # Launch the Gradio interface with queue enabled and sharing activated.
        # prevent_thread_lock lets us mount the auxiliary app on demo.app before
        # blocking: launch() rebuilds demo.app, so the mount has to come after it.
        demo.queue().launch(share=True, prevent_thread_lock=True)

        # Serve the embedding plot/images from this same server (same origin),
        # instead of a separate gradio.live tunnel that goes stale on restarts.
        mount_aux_server(demo)

        demo.block_thread()
    except KeyboardInterrupt:
        # Handle clean shutdown when user presses Ctrl+C
        print("Server closed")
        session_manager.stop_cleanup_thread()
