import gradio as gr
import requests
import json
import base64
from PIL import Image, ImageDraw
import io
import numpy as np

class OpenRouterClient:
    def __init__(self, api_key, base_url="https://openrouter.ai/api/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def encode_image(self, image):
        """Encode PIL Image to base64"""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    
    def get_pixel_coordinates(self, image, model, prompt):
        """Get pixel coordinates from specified model"""
        try:
            image_base64 = self.encode_image(image)
            
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_base64
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 100,
                "temperature": 0.1
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                # Extract coordinates from response - try multiple patterns
                import re
                
                # Try different coordinate patterns
                patterns = [
                    r'\[(\d+),\s*(\d+)\]',           # [x, y]
                    r'\((\d+),\s*(\d+)\)',           # (x, y)
                    r'(\d+),\s*(\d+)',               # x, y
                    r'x:\s*(\d+).*?y:\s*(\d+)',      # x: 123 ... y: 456
                    r'coordinates?:\s*\[?(\d+),\s*(\d+)\]?',  # coordinates: [x, y]
                    r'pixel.*?(\d+).*?(\d+)',        # pixel coordinates 123 456
                    r'click.*?(\d+).*?(\d+)',        # click at 123 456
                ]
                
                x, y = None, None
                matched_pattern = None
                
                for i, pattern in enumerate(patterns):
                    matches = re.search(pattern, content, re.IGNORECASE)
                    if matches:
                        try:
                            x, y = int(matches.group(1)), int(matches.group(2))
                            matched_pattern = f"Pattern {i+1}: {pattern}"
                            break
                        except (ValueError, IndexError):
                            continue
                
                if x is not None and y is not None:
                    return x, y, f"Found coordinates using {matched_pattern}\n\nFull response: {content}"
                else:
                    return None, None, f"Could not parse coordinates from response.\n\nTried {len(patterns)} patterns.\n\nFull response: {content}"
            else:
                return None, None, f"API Error: {response.status_code} - {response.text}"
                
        except Exception as e:
            return None, None, f"Error: {str(e)}"

def draw_red_dot(image, x, y, dot_size=20):
    """Draw a red dot on the image at specified coordinates"""
    if image is None:
        return None
    
    # Create a copy of the image
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    # Get image dimensions for validation
    img_width, img_height = img_copy.size
    
    # Clamp coordinates to image bounds
    x = max(0, min(x, img_width - 1))
    y = max(0, min(y, img_height - 1))
    
    # Draw a larger, more visible red circle
    left = x - dot_size // 2
    top = y - dot_size // 2
    right = x + dot_size // 2
    bottom = y + dot_size // 2
    
    # Draw filled circle with thick outline
    draw.ellipse([left, top, right, bottom], fill="red", outline="darkred", width=4)
    
    # Add a small crosshair for precise positioning
    crosshair_size = 5
    draw.line([x - crosshair_size, y, x + crosshair_size, y], fill="white", width=2)
    draw.line([x, y - crosshair_size, x, y + crosshair_size], fill="white", width=2)
    
    return img_copy

def process_image(image, api_key, base_url, model, user_prompt):
    """Process the image and return coordinates with marked image"""
    if image is None:
        return None, "Please upload an image first."
    
    if not api_key:
        return None, "Please provide an OpenRouter API key."
    
    if not model or not model.strip():
        return None, "Please enter a model ID."
    
    if not user_prompt or not user_prompt.strip():
        return None, "Please enter a prompt describing what to find."
    
    # Always extend the user prompt to ensure array format response
    extended_prompt = f"{user_prompt.strip()}\n\nIMPORTANT: You must respond with ONLY the pixel coordinates in the exact format [x,y] where x and y are numbers. Do not include any other text or explanation."
    
    # Initialize OpenRouter client
    client = OpenRouterClient(api_key, base_url)
    
    # Get pixel coordinates
    x, y, response_text = client.get_pixel_coordinates(image, model.strip(), extended_prompt)
    
    if x is not None and y is not None:
        # Get image dimensions for debugging
        img_width, img_height = image.size
        
        # Validate coordinates
        coord_valid = 0 <= x < img_width and 0 <= y < img_height
        coord_status = "✅ Valid" if coord_valid else f"⚠️ Out of bounds (image size: {img_width}x{img_height})"
        
        # Draw red dot on the image
        marked_image = draw_red_dot(image, x, y)
        
        result_text = f"""Coordinates: [{x}, {y}]
Image size: {img_width} x {img_height}
Coordinate status: {coord_status}
Model: {model}

User prompt: {user_prompt}
Extended prompt: {extended_prompt}

Raw model response:
{response_text}"""
        
        return marked_image, result_text
    else:
        return image, f"Failed to get coordinates from {model}: {response_text}"

# Create the Gradio interface
with gr.Blocks(title="OpenRouter Vision Pixel Detector", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎯 OpenRouter Vision Pixel Detector
    
    Upload an image and let AI models detect and mark pixel coordinates on objects.
    Enter any OpenRouter model ID to use for vision analysis.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Configuration")
            api_key_input = gr.Textbox(
                label="OpenRouter API Key",
                type="password",
                placeholder="Enter your OpenRouter API key",
                info="Get your API key from https://openrouter.ai/"
            )
            
            base_url_input = gr.Textbox(
                label="OpenRouter Base URL",
                value="https://openrouter.ai/api/v1",
                placeholder="https://openrouter.ai/api/v1"
            )
            
            model_input = gr.Textbox(
                label="Model ID",
                value="google/gemini-2.5-flash-preview-05-20",
                placeholder="e.g., anthropic/claude-3.5-sonnet",
                info="Enter any OpenRouter model ID that supports vision"
            )
            
            custom_prompt_input = gr.Textbox(
                label="Prompt (Required)",
                placeholder="e.g., 'Click on the person's face' or 'Find the red car'",
                lines=2,
                info="Describe what you want to find in the image"
            )
            
            process_btn = gr.Button("🔍 Analyze Image", variant="primary", size="lg")
            
        with gr.Column(scale=2):
            gr.Markdown("### Image Processing")
            with gr.Row():
                input_image = gr.Image(
                    label="Input Image",
                    type="pil",
                    height=400
                )
                output_image = gr.Image(
                    label="Output with Red Dot",
                    type="pil",
                    height=400
                )
            
            result_text = gr.Textbox(
                label="Results",
                lines=5,
                max_lines=10,
                interactive=False
            )
    
    # Event handlers
    process_btn.click(
        fn=process_image,
        inputs=[input_image, api_key_input, base_url_input, model_input, custom_prompt_input],
        outputs=[output_image, result_text]
    )
    
    # Auto-process when image is uploaded (if API key is provided)
    input_image.change(
        fn=lambda img, key, url, model, prompt: (
            process_image(img, key, url, model, prompt) 
            if key else (None, "Please provide API key first")
        ),
        inputs=[input_image, api_key_input, base_url_input, model_input, custom_prompt_input],
        outputs=[output_image, result_text]
    )
    
    gr.Markdown("""
    ### 📋 Instructions:
    1. **Get an API key** from [OpenRouter](https://openrouter.ai/) and paste it above
    2. **Enter a model ID** (default: UI-TARS-72B for computer vision tasks)
    3. **Enter a prompt** describing what you want to find in the image (required)
    4. **Upload an image** using the input area
    5. **Click Analyze** or wait for auto-processing
    6. **View results** - the output image will show a red dot at the detected coordinates
    
    ### 🤖 Popular Vision Model IDs:
    - `bytedance-research/ui-tars-72b` - Specialized for UI and computer vision tasks
    - `anthropic/claude-3.5-sonnet` - Excellent reasoning and vision capabilities
    - `google/gemini-2.5-flash-preview` - Fast and accurate multimodal processing
    - `openai/gpt-4o` - OpenAI's flagship vision model
    - `qwen/qwen3-32b` - Strong open-source vision model
    - `meta-llama/llama-3.2-90b-vision-instruct` - Meta's large vision model
    
    ### 💡 Tips:
    - **Required prompt**: You must always enter a prompt describing what to find
    - **Automatic extension**: Your prompt will be automatically extended to ensure array format response
    - Different models may interpret images differently
    - Some models work better with specific types of images
    - Try multiple models to compare results
    - Use specific prompts for better targeting (e.g., "Find the red button")
    """)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )