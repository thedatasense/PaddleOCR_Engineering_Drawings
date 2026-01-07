#!/usr/bin/env python3
"""
Engineering Drawing Information Extractor

Extracts material information, part numbers, and other metadata from
engineering drawing PDFs using a two-stage AI pipeline:
1. PaddleOCR-VL for OCR (text extraction from images)
2. Qwen3-0.6B for structured information extraction

Author: Binesh Kumar
Contact: contact@bineshkumar.me
Repository: https://github.com/thedatasense/PaddleOCR_Engineering_Drawings

License: MIT
"""

import os
import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
import pdf2image


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_PDF_PATH = "drawings/3814200A00.PDF"
DEFAULT_OUTPUT_DIR = "extraction_output"
OCR_MODEL_PATH = "PaddlePaddle/PaddleOCR-VL"
LLM_MODEL_PATH = "Qwen/Qwen3-0.6B"

# Device setup - prioritize MPS (Mac Silicon), then CUDA, then CPU
def get_device():
    """Detect the best available device."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = get_device()


def get_torch_dtype():
    """Get appropriate dtype for the device."""
    if DEVICE == "cuda":
        return torch.bfloat16
    elif DEVICE == "mps":
        return torch.float16  # MPS works better with float16
    return torch.float32


# =============================================================================
# Model Loading
# =============================================================================

def load_ocr_model():
    """Load PaddleOCR-VL model for OCR."""
    print(f"Loading PaddleOCR-VL model for OCR on {DEVICE}...")

    # Patch the check_model_inputs decorator compatibility issue
    import transformers.utils.generic as generic_utils
    original_check_model_inputs = generic_utils.check_model_inputs

    def patched_check_model_inputs(func_or_flag=True):
        if callable(func_or_flag):
            return original_check_model_inputs()(func_or_flag)
        return original_check_model_inputs(func_or_flag)

    generic_utils.check_model_inputs = patched_check_model_inputs

    model = AutoModelForCausalLM.from_pretrained(
        OCR_MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=get_torch_dtype()
    ).to(DEVICE).eval()

    processor = AutoProcessor.from_pretrained(OCR_MODEL_PATH, trust_remote_code=True)

    print("OCR model loaded!")
    return model, processor


def load_extraction_llm():
    """Load small LLM for structured extraction."""
    print(f"Loading extraction LLM: {LLM_MODEL_PATH} on {DEVICE}...")

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_PATH,
        torch_dtype=get_torch_dtype()
    ).to(DEVICE).eval()

    print("Extraction LLM loaded!")
    return model, tokenizer


# =============================================================================
# OCR Functions
# =============================================================================

def convert_pdf_to_images(pdf_path, dpi=200):
    """Convert PDF pages to images."""
    print(f"Converting PDF to images: {pdf_path}")
    images = pdf2image.convert_from_path(pdf_path, dpi=dpi)
    print(f"Converted {len(images)} page(s)")
    return images


def perform_ocr(image, model, processor):
    """Perform OCR on an image using PaddleOCR-VL."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "OCR:"},
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)
    pixel_values = inputs["pixel_values"].to(DEVICE)
    image_grid_thw = inputs["image_grid_thw"].to(DEVICE)

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            max_new_tokens=2048,
            do_sample=False,
            use_cache=True
        )

    result = processor.batch_decode(outputs, skip_special_tokens=True)[0]

    # Clean up the response
    if "Assistant:" in result:
        result = result.split("Assistant:")[-1].strip()

    return result


# =============================================================================
# LLM Extraction
# =============================================================================

def extract_info_with_llm(ocr_text, llm_model, tokenizer):
    """
    Use a small LLM to extract structured information from OCR text.

    This function uses a generalized prompt that works with various
    engineering drawing formats (bilingual, single language, etc.)
    """

    extraction_prompt = f"""Extract these fields from the engineering drawing text below:

TEXT:
{ocr_text}

Find and extract:
1. part_number - The 7-digit number (like 3814200) after "CODE" or "Part Number"
2. drawing_number - The code after "n° plan" or "Drawing Number" (like 4095700.M00.027PI1/2)
3. material - After "Matière" or "Material" (e.g. Polycarbonate makrolon cristal ref:2458)
4. finish - After "Aspect" (e.g. poli, polish)
5. description - After "Désignation" or "Description" - the part name like "CAPOT INTERRUPTEUR / SWITCH COVER"
6. company - Company name only (e.g. COVIDIEN), not the address
7. product - After "Appareil" or "Product" - device name like "LEGENDAIR XL2 US"

Reply with exactly this format:
part_number: <extracted value>
drawing_number: <extracted value>
material: <extracted value>
finish: <extracted value>
description: <extracted value>
company: <extracted value>
product: <extracted value>"""

    messages = [{"role": "user", "content": extraction_prompt}]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False  # Disable thinking for faster inference
    )

    inputs = tokenizer([text], return_tensors="pt").to(DEVICE)

    with torch.inference_mode():
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )

    # Parse the response into a dictionary
    extracted = {}
    for line in response.strip().split('\n'):
        if ':' in line:
            parts = line.split(':', 1)
            if len(parts) == 2:
                key = parts[0].strip().lower().replace(' ', '_').replace('-', '_')
                value = parts[1].strip()
                key = key.lstrip('_').lstrip('-')
                # Skip placeholder values
                if value and value.lower() not in ["not found", "n/a", "none", "[value]", ""]:
                    extracted[key] = value

    return extracted


# =============================================================================
# Main Processing
# =============================================================================

def process_drawing(pdf_path, output_dir):
    """
    Process an engineering drawing PDF and extract information.

    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save output files

    Returns:
        List of extraction results for each page
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Convert PDF to images
    images = convert_pdf_to_images(pdf_path)

    # Save images for reference
    for i, img in enumerate(images):
        img_path = os.path.join(output_dir, f"page_{i+1}.png")
        img.save(img_path)
        print(f"Saved: {img_path}")

    # Load models
    ocr_model, ocr_processor = load_ocr_model()
    llm_model, llm_tokenizer = load_extraction_llm()

    # Process each page
    all_results = []

    for i, image in enumerate(images):
        print(f"\n{'='*60}")
        print(f"Processing page {i+1}...")
        print('='*60)

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Step 1: Perform OCR
        print("\nStep 1: Performing OCR with PaddleOCR-VL...")
        ocr_text = perform_ocr(image, ocr_model, ocr_processor)

        print(f"\nRAW OCR OUTPUT (Page {i+1}):")
        print("-" * 40)
        print(ocr_text[:500] + "..." if len(ocr_text) > 500 else ocr_text)
        print("-" * 40)

        # Step 2: Extract structured information (for main drawing pages)
        extracted_info = {}
        if i == 0:  # Main drawing page
            print("\nStep 2: Extracting structured info with Qwen3-0.6B...")
            extracted_info = extract_info_with_llm(ocr_text, llm_model, llm_tokenizer)

            print(f"\nEXTRACTED INFORMATION (Page {i+1}):")
            print("-" * 40)
            for key, value in extracted_info.items():
                print(f"  {key}: {value}")
            print("-" * 40)

        page_result = {
            "page": i + 1,
            "raw_ocr_text": ocr_text,
            "extracted_info": extracted_info
        }
        all_results.append(page_result)

    # Save results
    output_json_path = os.path.join(output_dir, "extraction_results.json")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_json_path}")

    output_text_path = os.path.join(output_dir, "raw_ocr_text.txt")
    with open(output_text_path, 'w', encoding='utf-8') as f:
        for result in all_results:
            f.write(f"=== Page {result['page']} ===\n")
            f.write(result['raw_ocr_text'])
            f.write("\n\n")
    print(f"Raw OCR text saved to: {output_text_path}")

    return all_results


def print_summary(results):
    """Print a summary of extracted data."""
    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)

    if results and "extracted_info" in results[0]:
        info = results[0]["extracted_info"]
        fields = [
            ("Part Number", "part_number"),
            ("Drawing Number", "drawing_number"),
            ("Material", "material"),
            ("Finish", "finish"),
            ("Description", "description"),
            ("Company", "company"),
            ("Product", "product"),
        ]
        for label, key in fields:
            value = info.get(key, "N/A")
            print(f"  {label}: {value}")

    print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract information from engineering drawing PDFs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_drawing_info.py
  python extract_drawing_info.py --pdf drawing.pdf --output results/

Author: Binesh Kumar (contact@bineshkumar.me)
        """
    )
    parser.add_argument(
        "--pdf", "-p",
        default=DEFAULT_PDF_PATH,
        help=f"Path to the PDF file (default: {DEFAULT_PDF_PATH})"
    )
    parser.add_argument(
        "--output", "-o",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Engineering Drawing Information Extractor")
    print("Author: Binesh Kumar (contact@bineshkumar.me)")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"PDF: {args.pdf}")
    print(f"Output: {args.output}")
    print("=" * 60)

    results = process_drawing(args.pdf, args.output)
    print_summary(results)

    print("\nExtraction complete!")
    return results


if __name__ == "__main__":
    main()
