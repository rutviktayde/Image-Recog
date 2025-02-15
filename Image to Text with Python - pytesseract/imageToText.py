from flask import Flask, request, render_template_string
import pytesseract as tess
from PIL import Image, ImageEnhance, ImageFilter
import os
import io

app = Flask(__name__)
tess.pytesseract.tesseract_cmd = r"C:\Users\HP\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Enhanced OCR Interface</title>
    <style>
        body { 
            margin: 0;
            padding: 20px;
            height: 100vh;
            background: #f0f0f0;
            display: flex;
            gap: 20px;
        }
        .window {
            flex: 1;
            height: calc(100% - 40px);
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            overflow-y: auto;
        }
        #preview {
            max-width: 100%;
            max-height: 60vh;
            margin: 20px 0;
        }
        .upload-btn {
            padding: 12px 24px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            margin-bottom: 15px;
        }
        .tips {
            color: #666;
            font-size: 0.9em;
            margin: 10px 0;
        }
        #result {
            white-space: pre-wrap;
            font-family: monospace;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 6px;
            margin-top: 20px;
        }
        .error {
            color: #dc3545;
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <div class="window">
        <input type="file" id="upload" accept="image/*" hidden>
        <button class="upload-btn" onclick="document.getElementById('upload').click()">
            Upload Image
        </button>
        <div class="tips">
            <strong>Tips for better recognition:</strong><br>
            • Use high-contrast images<br>
            • Ensure text is horizontal<br>
            • Avoid curved text layouts<br>
            • Use minimum 300dpi resolution
        </div>
        <img id="preview">
        <div id="result"></div>
        <div class="error" id="error"></div>
    </div>
    
    <div class="window">
        <!-- Right window remains blank -->
    </div>

    <script>
        const upload = document.getElementById('upload');
        const preview = document.getElementById('preview');
        const result = document.getElementById('result');
        const error = document.getElementById('error');

        upload.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            const formData = new FormData();
            formData.append('image', file);

            try {
                error.textContent = '';
                result.textContent = 'Processing...';
                
                const response = await fetch('/ocr', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    const errorText = await response.text();
                    throw new Error(errorText);
                }
                
                const text = await response.text();
                result.textContent = text;
                
                // Show preview
                const reader = new FileReader();
                reader.onload = (e) => preview.src = e.target.result;
                reader.readAsDataURL(file);
                
            } catch (error) {
                result.textContent = '';
                error.textContent = 'Error: ' + error.message;
                console.error('Error:', error);
            }
        });
    </script>
</body>
</html>
'''

def preprocess_image(img):
    """Enhance image for better OCR results"""
    # Convert to grayscale
    img = img.convert('L')
    
    # Resize to improve resolution
    width, height = img.size
    img = img.resize((width*2, height*2), Image.Resampling.LANCZOS)
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)
    
    # Apply sharpening
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(2.0)
    
    # Apply thresholding
    img = img.point(lambda x: 0 if x < 140 else 255)
    
    # Remove noise
    img = img.filter(ImageFilter.MedianFilter(size=3))
    
    return img

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/ocr', methods=['POST'])
def ocr():
    try:
        if 'image' not in request.files:
            return 'No image uploaded', 400
        
        file = request.files['image']
        if file.filename == '':
            return 'No selected file', 400
            
        # Open and preprocess image
        img = Image.open(io.BytesIO(file.read()))
        img = preprocess_image(img)
        
        # Custom OCR configuration
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?()[]{}@#$%^&*_+-=/:;'
        
        text = tess.image_to_string(img, config=custom_config)
        
        if not text.strip():
            return 'No text detected. Try a clearer image.', 400
            
        return text
    
    except Exception as e:
        return f'OCR processing failed: {str(e)}', 500

if __name__ == '__main__':
    app.run(debug=True)