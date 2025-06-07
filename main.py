from fastapi import FastAPI, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import Optional

app = FastAPI()

# Configure CORS (allow all for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path to your models directory
MODELS_DIR = Path("./VR_Storage")
print(MODELS_DIR)

PORT = 8000

# Mount static files (so thumbnails and models can be accessed directly)
app.mount("/static", StaticFiles(directory=MODELS_DIR), name="static")

@app.get("/api/models", response_model=List[Dict[str, Any]])
def get_models():
    models = []
    
    if not MODELS_DIR.exists():
        raise HTTPException(status_code=404, detail="Models directory not found")
    
    for model_dir in MODELS_DIR.iterdir():
        if model_dir.is_dir():
            metadata_path = model_dir / "metadata.json"
            
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        
                    # Get the most recent modified time in the directory
                    last_updated = max(
                        (f.stat().st_mtime for f in model_dir.glob('*') if f.is_file()),
                        default=os.path.getmtime(model_dir)
                    )
                    
                    models.append({
                        "id": model_dir.name,
                        "name": metadata.get("name", model_dir.name),
                        "description": metadata.get("description", "No description available"),
                        "lastUpdated": datetime.fromtimestamp(last_updated).isoformat(),
                        "thumbnail": f"http://127.0.0.1:{PORT}/static/{model_dir.name}/{metadata.get('thumbnail', 'thumbnail.jpg')}",
                        "modelPath": f"http://127.0.0.1:{PORT}/static/{model_dir.name}/{metadata.get('modelFile', 'model.glb')}"
                    })
                except json.JSONDecodeError:
                    continue
    
    return models


MODELS_DIR.mkdir(exist_ok=True)

@app.post("/api/models")
async def create_model(
    name: str = Form(...),
    description: str = Form(""),
    thumbnail: Optional[UploadFile] = File(None),
    model: UploadFile = File(...)
):
    # Sanitize model name for folder name
    folder_name = "".join(c if c.isalnum() else "_" for c in name.lower())
    model_dir = MODELS_DIR / folder_name
    model_dir.mkdir(exist_ok=True)
    
    # Save 3D model file
    model_ext = Path(model.filename).suffix.lower()
    model_filename = f"model{model_ext}"
    model_path = model_dir / model_filename
    with open(model_path, "wb") as buffer:
        buffer.write(await model.read())
    
    # Save thumbnail (default to thumbnail.jpg if not provided)
    thumbnail_filename = "thumbnail.jpg"
    if thumbnail:
        thumbnail_ext = Path(thumbnail.filename).suffix.lower()
        thumbnail_filename = f"thumbnail{thumbnail_ext}"
        thumbnail_path = model_dir / thumbnail_filename
        with open(thumbnail_path, "wb") as buffer:
            buffer.write(await thumbnail.read())
    
    # Create metadata.json
    metadata = {
        "name": name,
        "description": description,
        "thumbnail": thumbnail_filename,
        "modelFile": model_filename,
        "createdAt": datetime.now().isoformat(),
        "lastUpdated": datetime.now().isoformat()
    }
    
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    return {
        "success": True,
        "message": "Model created successfully",
        "modelId": folder_name
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)