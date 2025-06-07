from fastapi import FastAPI, Form, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import shutil
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

app = FastAPI()

# Configure CORS (allow all for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MODELS_DIR = Path("./VR_Storage")
PORT = 8000
BASE_URL = f"http://127.0.0.1:{PORT}"

# Ensure models directory exists
MODELS_DIR.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=MODELS_DIR), name="static")

def generate_model_id() -> str:
    return str(uuid.uuid4())

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
                    
                    # Get the most recent modified time
                    last_updated = max(
                        (f.stat().st_mtime for f in model_dir.glob('*') if f.is_file()),
                        default=os.path.getmtime(model_dir)
                    )
                    
                    models.append({
                        "id": metadata.get("id", model_dir.name),
                        "name": metadata.get("name", model_dir.name),
                        "description": metadata.get("description", "No description available"),
                        "lastUpdated": datetime.fromtimestamp(last_updated).isoformat(),
                        "thumbnail": f"{BASE_URL}/static/{model_dir.name}/{metadata.get('thumbnail', 'thumbnail.jpg')}",
                        "modelPath": f"{BASE_URL}/static/{model_dir.name}/{metadata.get('modelFile', 'model.glb')}",
                        "createdAt": metadata.get("createdAt", "")
                    })
                except json.JSONDecodeError:
                    continue
    
    return models

@app.post("/api/models")
async def create_model(
    name: str = Form(...),
    description: str = Form(""),
    thumbnail: Optional[UploadFile] = File(None),
    model: UploadFile = File(...)
):
    # Generate unique ID
    model_id = generate_model_id()
    
    # Create model directory using ID as folder name
    model_dir = MODELS_DIR / model_id
    model_dir.mkdir(exist_ok=True)
    
    # Save 3D model file
    model_ext = Path(model.filename).suffix.lower()
    model_filename = f"model{model_ext}"
    model_path = model_dir / model_filename
    with open(model_path, "wb") as buffer:
        buffer.write(await model.read())
    
    # Save thumbnail
    thumbnail_filename = "thumbnail.jpg"
    if thumbnail:
        thumbnail_ext = Path(thumbnail.filename).suffix.lower()
        thumbnail_filename = f"thumbnail{thumbnail_ext}"
        thumbnail_path = model_dir / thumbnail_filename
        with open(thumbnail_path, "wb") as buffer:
            buffer.write(await thumbnail.read())
    
    # Create metadata.json with ID
    metadata = {
        "id": model_id,
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
        "modelId": model_id,
        "metadata": metadata
    }

@app.delete("/api/models/{model_id}")
async def delete_model(model_id: str):
    model_dir = MODELS_DIR / model_id
    
    if not model_dir.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        # Remove the entire directory
        shutil.rmtree(model_dir)
        return {
            "success": True,
            "message": "Model deleted successfully"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to delete model: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)