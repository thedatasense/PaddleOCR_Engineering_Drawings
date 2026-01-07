# Engineering Drawing Information Extractor

Extract material information, part numbers, and other metadata from engineering drawing PDFs using a two-stage AI pipeline.

## Overview

This tool uses:
1. **[PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL)** - A 0.9B vision-language model for OCR (text extraction from images)
2. **[Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)** - A small 600M parameter LLM for structured information extraction

No complex regex patterns - the extraction is done entirely by the LLM, making it generalizable to different drawing formats.

## Features

- Extracts key information from engineering drawings:
  - Part Number
  - Drawing Number
  - Material Specification
  - Surface Finish
  - Part Description
  - Company Name
  - Product Name
- Supports bilingual drawings (French/English)
- GPU acceleration (CUDA, Apple Silicon MPS)
- Outputs structured JSON and raw OCR text
- Command-line interface with customizable options

## Installation

### Prerequisites

1. **Python 3.10+**

2. **Poppler** (required for PDF to image conversion):
   ```bash
   # macOS
   brew install poppler

   # Ubuntu/Debian
   sudo apt-get install poppler-utils

   # Windows
   # Download from https://github.com/oschwartz10612/poppler-windows/releases
   # Add to PATH
   ```

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/thedatasense/PaddleOCR_Engineering_Drawings.git
cd PaddleOCR_Engineering_Drawings

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python extract_drawing_info.py
```

This will process the default PDF (`drawings/3814200A00.PDF`) and save results to `extraction_output/`.

### Custom PDF

```bash
python extract_drawing_info.py --pdf path/to/your/drawing.pdf --output results/
```

### Command Line Options

```
-p, --pdf      Path to the PDF file (default: drawings/3814200A00.PDF)
-o, --output   Output directory (default: extraction_output)
```

## Output

The tool generates:

1. **`extraction_results.json`** - Structured extraction results
2. **`raw_ocr_text.txt`** - Raw OCR output from each page
3. **`page_N.png`** - Converted images of each PDF page

### Example Output

```json
{
  "page": 1,
  "raw_ocr_text": "...",
  "extracted_info": {
    "part_number": "3814200",
    "drawing_number": "4095700.M00.027PI1/2",
    "material": "Polycarbonate (makrolon cristal ref:2458)",
    "finish": "poli",
    "description": "CAPOT INTERRUPTEUR / SWITCH COVER",
    "company": "COVIDIEN",
    "product": "LEGENDAIR XL2 US"
  }
}
```

## Hardware Requirements

- **Minimum**: 8GB RAM (CPU mode)
- **Recommended**: 16GB RAM with GPU
- **GPU Support**:
  - NVIDIA CUDA
  - Apple Silicon (MPS)

The tool automatically detects and uses the best available device.

## Sample Drawing

The `drawings/` directory contains a sample engineering drawing (`3814200A00.PDF`) from Medtronic/Covidien for testing purposes. This drawing is provided under Medtronic's license agreement included in the repository.

## How It Works

```
PDF → Images → PaddleOCR-VL (OCR) → Qwen3-0.6B (Extraction) → Structured JSON
     (pdf2image)   (Vision-Language)      (Text LLM)
```

1. **PDF Conversion**: The PDF is converted to high-resolution images using `pdf2image`
2. **OCR**: PaddleOCR-VL extracts all text from the drawing images
3. **Extraction**: Qwen3-0.6B analyzes the OCR text and extracts structured fields
4. **Output**: Results are saved as JSON and text files

## Customization

### Adding New Fields

To extract additional fields, modify the `extraction_prompt` in `extract_info_with_llm()`:

```python
extraction_prompt = f"""Extract these fields from the engineering drawing text below:

TEXT:
{ocr_text}

Find and extract:
1. part_number - ...
2. drawing_number - ...
3. your_new_field - Description of where to find it
...
"""
```

### Using Different Models

You can change the models by modifying the configuration:

```python
OCR_MODEL_PATH = "PaddlePaddle/PaddleOCR-VL"  # Vision-language OCR
LLM_MODEL_PATH = "Qwen/Qwen3-0.6B"            # Text extraction LLM
```

## Troubleshooting

### "PDFInfoNotInstalledError"
Install Poppler (see Prerequisites above).

### Out of Memory
- Use CPU mode by setting `DEVICE = "cpu"` in the script
- Close other applications to free up RAM

### Slow Performance
- Ensure GPU is being used (check "Device: mps" or "Device: cuda" in output)
- Reduce image DPI in `convert_pdf_to_images(pdf_path, dpi=150)`

## License

MIT License

## Author

**Binesh Kumar**
- Email: contact@bineshkumar.me
- GitHub: [@thedatasense](https://github.com/thedatasense)

## Acknowledgments

- [PaddlePaddle](https://github.com/PaddlePaddle/PaddleOCR) for PaddleOCR-VL
- [Qwen](https://github.com/QwenLM/Qwen) for Qwen3-0.6B
- Medtronic PB10 Opensource ventilator
