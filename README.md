This whole repo is vibe coded with Sonnet 4.0.

# 🎯 OpenRouter Vision Pixel Detector

A Gradio application that uses OpenRouter and various AI vision models to detect and mark pixel coordinates on uploaded images. Choose from popular vision models or use any custom OpenRouter model ID.

## Features

- 🖼️ **Image Upload**: Upload any image for analysis
- 🤖 **Multiple AI Models**: Choose from popular vision models or enter any OpenRouter model ID
- 🔴 **Visual Marking**: Automatically marks detected points with red dots
- ⚙️ **Customizable**: Custom prompts for specific object detection
- 🔗 **OpenRouter Integration**: Configurable API endpoint and authentication
- 🎯 **Model Comparison**: Test different models on the same image

## Quick Start

### Using Docker Compose (Recommended)

1. **Clone and navigate to the project:**
   ```bash
   cd gradio-computer-use-model-test
   ```

2. **Start the application:**
   ```bash
   docker-compose up
   ```

3. **Open your browser:**
   Navigate to `http://localhost:7860`

### Manual Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   python app.py
   ```

## Configuration

1. **Get an OpenRouter API key:**
   - Visit [OpenRouter](https://openrouter.ai/)
   - Sign up and get your API key

2. **Configure the application:**
   - Enter your API key in the interface
   - Select a model from the dropdown or enter a custom model ID
   - Optionally modify the base URL if needed
   - Customize prompts for specific detection tasks

## Usage

1. **Upload an image** using the input area
2. **Enter your OpenRouter API key**
3. **Select a model** from the dropdown or enter a custom model ID
4. **Optionally customize the prompt** (e.g., "Click on the person's face")
5. **Click "Analyze Image"** or wait for auto-processing
6. **View the results** - coordinates and marked image with red dot

## Popular Vision Models

### Included in Dropdown:
- **UI-TARS-72B** (`meta-llama/llama-3.2-11b-vision-instruct:free`): Specialized for UI and computer vision tasks
- **Claude 3.5 Sonnet** (`anthropic/claude-3.5-sonnet`): Excellent reasoning and vision capabilities
- **Gemini 2.5 Flash** (`google/gemini-2.5-flash-preview`): Fast and accurate multimodal processing
- **Gemini Flash 1.5** (`google/gemini-flash-1.5`): High-volume, cost-effective vision model
- **GPT-4o** (`openai/gpt-4o`): OpenAI's flagship vision model
- **GPT-4o Mini** (`openai/gpt-4o-mini`): Faster, more affordable GPT-4o variant
- **Qwen3-32B** (`qwen/qwen3-32b`): Strong open-source vision model
- **Llama 3.2 90B Vision** (`meta-llama/llama-3.2-90b-vision-instruct`): Meta's large vision model
- **Claude 3 Haiku** (`anthropic/claude-3-haiku`): Fast and efficient vision model
- **Gemini Pro Vision** (`google/gemini-pro-vision`): Google's professional vision model

### Custom Models:
You can use **any OpenRouter model** by entering its ID in the custom model field. Popular options include:
- `anthropic/claude-3-opus`
- `mistralai/pixtral-12b`
- `qwen/qwen2-vl-72b-instruct`
- `microsoft/phi-3.5-vision-instruct`

## Example Prompts

- "Click on the most prominent object"
- "Find the person's face"
- "Locate the red car"
- "Click on the submit button"
- "Find the text input field"
- "Point to the main heading"
- "Identify the search box"

## Technical Details

- **Framework**: Gradio 4.0+
- **Image Processing**: Pillow (PIL)
- **API Client**: Custom OpenRouter integration
- **Container**: Python 3.11 slim base image
- **Port**: 7860 (configurable)
- **Supported Formats**: PNG, JPEG, WebP, and other PIL-supported formats

## Model Comparison Tips

- **UI-TARS**: Best for UI elements and interface detection
- **Claude 3.5 Sonnet**: Excellent for complex reasoning and detailed analysis
- **Gemini Models**: Fast processing and good general vision capabilities
- **GPT-4o**: Strong overall performance across various image types
- **Open Source Models**: Cost-effective alternatives with good performance

## Troubleshooting

- **API Errors**: Verify your OpenRouter API key and credit balance
- **No Coordinates Found**: Try a more specific prompt or different model
- **Connection Issues**: Check your internet connection and OpenRouter status
- **Model Not Found**: Ensure the custom model ID is correct and available on OpenRouter
- **Rate Limiting**: Some models may have usage limits; try a different model

## License

MIT License - feel free to modify and distribute as needed. 