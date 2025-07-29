import warnings
import time
import os
import torch

# Suppress all warnings
warnings.filterwarnings("ignore")

# Force CPU usage and offline mode (avoid CUDA NMS issues and online downloads)
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['YOLO_CONFIG_DIR'] = '/tmp'
os.environ['ULTRALYTICS_OFFLINE'] = '1'
torch.cuda.is_available = lambda: False

from ultralytics import YOLO
import fitz
from PIL import Image
import cv2
import numpy as np
import easyocr
import json
from io import BytesIO
import sys
import re
from pathlib import Path
from typing import Optional

class DockerOutlineExtractor:
    def __init__(self, model_path="/model/yolov11x_best.pt"):
        """Initialize extractor with Docker-compatible paths and robust error handling"""
        print("Loading YOLO model...")
        
        # Verify YOLO model file exists and is valid
        model_file = Path(model_path)
        if not model_file.exists():
            print(f"❌ YOLO model file not found: {model_path}")
            print("Available files in /model/:")
            model_dir = Path("/model")
            if model_dir.exists():
                for f in model_dir.glob("*"):
                    print(f"  {f.name} ({f.stat().st_size:,} bytes)")
            else:
                print("  /model directory not found!")
            raise FileNotFoundError(f"YOLO model not found: {model_path}")
        
        if model_file.stat().st_size < 1000000:  # Less than 1MB indicates corrupted file
            print(f"❌ YOLO model file too small: {model_file.stat().st_size:,} bytes")
            raise ValueError(f"YOLO model file appears corrupted: {model_path}")
        
        print(f"✅ Found YOLO model: {model_path} ({model_file.stat().st_size:,} bytes)")
        
        try:
            # Load YOLO model with offline enforcement
            self.model = YOLO(model_path)
            self.model.to('cpu')
            print("✅ YOLO model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading YOLO model: {e}")
            raise
        
        print("Initializing OCR...")
        
        # Docker-compatible EasyOCR model path with better handling
        easyocr_models_path = Path("/model/easyocr")
        
        # Check for required EasyOCR model files
        required_models = [
            easyocr_models_path / "craft_mlt_25k.pth",
            easyocr_models_path / "latin_g2.pth"
        ]
        
        models_complete = all(
            model_file.exists() and model_file.stat().st_size > 1000000 
            for model_file in required_models
        )
        
        if models_complete:
            print(f"✅ Complete EasyOCR models found at: {easyocr_models_path}")
            try:
                self.ocr = easyocr.Reader(['en'], gpu=False, model_storage_directory=str(easyocr_models_path))
                print("✅ EasyOCR initialized with local models")
            except Exception as e:
                print(f"⚠️ Failed to use local EasyOCR models: {e}")
                print("Falling back to default EasyOCR (may fail in offline mode)")
                self.ocr = easyocr.Reader(['en'], gpu=False)
        else:
            print(f"⚠️ Incomplete EasyOCR models at {easyocr_models_path}")
            print("Missing or corrupted model files:")
            for model_file in required_models:
                if not model_file.exists():
                    print(f"  ❌ Missing: {model_file.name}")
                elif model_file.stat().st_size < 1000000:
                    print(f"  ❌ Too small: {model_file.name} ({model_file.stat().st_size:,} bytes)")
            
            # Try fallback (will likely fail in offline Docker mode)
            print("Attempting fallback to default EasyOCR location...")
            try:
                self.ocr = easyocr.Reader(['en'], gpu=False)
                print("✅ EasyOCR initialized with default location")
            except Exception as e:
                print(f"❌ EasyOCR initialization failed: {e}")
                raise RuntimeError("Cannot initialize EasyOCR. Ensure models are properly downloaded in Docker build.")
        
        self.target_classes = ["Title", "Section-header"]
        print("✅ All models loaded successfully")

    def pdf_to_images(self, pdf_path):
        """Convert PDF pages to images"""
        doc = fitz.open(pdf_path)
        images = []
        for i in range(len(doc)):
            page = doc[i]
            mat = fitz.Matrix(2.0, 2.0)
            try:
                pix = page.get_pixmap(matrix=mat)
            except AttributeError:
                pix = page.getPixmap(matrix=mat)
            
            img_data = pix.tobytes("png")
            pil_img = Image.open(BytesIO(img_data))
            cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            images.append(cv_img)
        doc.close()
        return images

    def extract_text(self, image, bbox):
        """Extract text from bounding box using OCR"""
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return ""
        try:
            results = self.ocr.readtext(crop)
            chunks = [r[1] for r in results if len(r) >= 3 and r[2] > 0.5]
            return " ".join(chunks).strip()
        except Exception as e:
            print(f"⚠️ OCR error for bbox {bbox}: {e}")
            return ""

    def assign_hierarchy(self, detected_text, bbox, area, rel_y_pos, page, first_title_found):
        """Assign hierarchy level with unlimited depth support"""
        text = detected_text.strip()
        if not text: 
            return None

        # Handle unlimited hierarchy depth based on numbering
        # Examples: "1." -> H1, "1.1." -> H2, "1.1.1." -> H3, "1.1.1.1." -> H4, etc.
        match = re.match(r'^(\d+(?:\.\d+)*)[.\s]', text)
        if match:
            numbering = match.group(1)
            depth = numbering.count('.') + 1
            return f"H{depth}"
        
        # Alternative numbering patterns
        # "1.1.1.1 Text" or "1.1.1.1. Text"
        match = re.match(r'^(\d+(?:\.\d+)+)[\s.]', text)
        if match:
            numbering = match.group(1)
            depth = numbering.count('.') + 1
            return f"H{depth}"

        # Fallback to position and size-based heuristics
        if area > 15000 and rel_y_pos < 0.20:
            return "H1"
        elif area > 6000:
            return "H2"
        else:
            return "H3"

    def get_outline(self, pdf_path):
        """Extract outline from a single PDF"""
        print(f"Processing: {Path(pdf_path).name}")
        
        try:
            images = self.pdf_to_images(pdf_path)
        except Exception as e:
            print(f"❌ Error converting PDF to images: {e}")
            raise
        
        outline = []
        title = None
        first_title_found = False

        for page_idx, img in enumerate(images, 1):
            try:
                # Run YOLO detection with explicit offline settings
                detections = self.model(img, conf=0.25, device='cpu', verbose=False)
                
                for result in detections:
                    if result.boxes is None: 
                        continue
                    
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        class_name = self.model.names[class_id]
                        
                        if class_name not in self.target_classes:
                            continue
                        
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        bbox = [int(x1), int(y1), int(x2), int(y2)]
                        
                        area = (x2 - x1) * (y2 - y1)
                        rel_y = y1 / img.shape[0] if img.shape[0] > 0 else 0.0

                        text = self.extract_text(img, bbox)
                        if not text: 
                            continue
                        text = text.strip()

                        if class_name == "Title":
                            if not first_title_found:
                                title = text
                                level = "H1"
                                first_title_found = True
                            else:
                                # Skip additional titles after the first one
                                continue 
                        else:
                            level = self.assign_hierarchy(text, bbox, area, rel_y, page_idx, first_title_found)
                            if not level: 
                                continue

                        # Skip titles found on pages after the first
                        if class_name == "Title" and page_idx > 1:
                            continue

                        outline.append({
                            "level": level,
                            "text": text,
                            "page": page_idx,
                        })
                        
            except Exception as e:
                print(f"⚠️ Error processing page {page_idx}: {e}")
                continue
        
        # Fallback title selection if no title was found
        if title is None and outline:
            for item in outline:
                if item["level"] in ["H1", "H2"]:
                    title = item["text"]
                    break
            if title is None:
                title = outline[0]["text"]

        return {"title": title or "(unknown)", "outline": outline}

    def save_json(self, data, output_path):
        """Save outline data to JSON file"""
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"❌ Error saving JSON to {output_path}: {e}")
            raise

    def process_all_pdfs(self, input_dir="/app/input", output_dir="/app/output"):
        """
        Batch process all PDFs in input directory - MAIN DOCKER FUNCTION
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Verify input directory exists
        if not input_path.exists():
            print(f"❌ Input directory not found: {input_dir}")
            print("Docker volume mapping may be incorrect.")
            return
        
        # Create output directory if it doesn't exist
        try:
            output_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"❌ Cannot create output directory {output_dir}: {e}")
            return
        
        # Find all PDF files in input directory
        pdf_files = list(input_path.glob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {input_dir}")
            print("Available files:")
            for file in input_path.glob("*"):
                print(f"  {file.name}")
            return
        
        print(f"Found {len(pdf_files)} PDF file(s) to process")
        
        total_start_time = time.time()
        successful_count = 0
        failed_count = 0
        
        for pdf_file in pdf_files:
            file_start_time = time.time()
            
            try:
                # Extract outline
                outline_data = self.get_outline(str(pdf_file))
                
                # Save to output directory with same name but .json extension
                output_file = output_path / (pdf_file.stem + ".json")
                self.save_json(outline_data, output_file)
                
                file_end_time = time.time()
                file_time = file_end_time - file_start_time
                
                print(f"✅ Saved: {output_file.name} ({file_time:.2f}s)")
                successful_count += 1
                
            except Exception as e:
                file_end_time = time.time()
                file_time = file_end_time - file_start_time
                print(f"❌ Error processing {pdf_file.name}: {str(e)} ({file_time:.2f}s)")
                failed_count += 1
                continue
        
        total_end_time = time.time()
        total_time = total_end_time - total_start_time
        
        # Format total time
        hours, rem = divmod(total_time, 3600)
        minutes, seconds = divmod(rem, 60)
        
        time_str = ""
        if hours > 0:
            time_str += f"{int(hours)}h "
        if minutes > 0:
            time_str += f"{int(minutes)}m "
        time_str += f"{seconds:.2f}s"
        
        print(f"\n📊 PROCESSING SUMMARY")
        print(f"{'='*50}")
        print(f"Total files: {len(pdf_files)}")
        print(f"Successful: {successful_count}")
        print(f"Failed: {failed_count}")
        print(f"Total processing time: {time_str}")
        print("Batch processing complete")

def main():
    """
    Main function for Docker container
    Automatically processes all PDFs in /app/input and outputs to /app/output
    """
    print("🚀 Document Outline Extractor Starting...")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    start_time = time.time()
    
    try:
        # Initialize extractor
        extractor = DockerOutlineExtractor()
        
        # Process all PDFs in batch mode
        extractor.process_all_pdfs()
        
    except Exception as e:
        print(f"💥 Fatal error during batch processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
