from fastapi import FastAPI, Form, HTTPException, UploadFile, File, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import shutil
import uuid
import time
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

# Custom StaticFiles class to disable caching
class NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

# Mount static files with no caching
app.mount("/static", NoCacheStaticFiles(directory=MODELS_DIR), name="static")

def generate_model_id() -> str:
    return str(uuid.uuid4())

@app.get("/api/models/search")
def search_models(query: str = ""):
    models = []
    cache_buster = int(time.time())  # Current timestamp for cache busting
    
    if not MODELS_DIR.exists():
        raise HTTPException(status_code=404, detail="Models directory not found")
    
    for model_dir in MODELS_DIR.iterdir():
        if model_dir.is_dir():
            metadata_path = model_dir / "metadata.json"
            
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Skip if query doesn't match name or description
                    if query.lower() not in metadata.get("name", "").lower() and \
                       query.lower() not in metadata.get("description", "").lower():
                        continue
                        
                    last_updated = max(
                        (f.stat().st_mtime for f in model_dir.glob('*') if f.is_file()),
                        default=os.path.getmtime(model_dir)
                    )
                    
                    # Add transformation matrix to the response (default identity matrix if not present)
                    transform = metadata.get("transform", [
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]
                    ])
                    
                    models.append({
                        "id": metadata.get("id", model_dir.name),
                        "name": metadata.get("name", model_dir.name),
                        "description": metadata.get("description", "No description available"),
                        "lastUpdated": datetime.fromtimestamp(last_updated).isoformat(),
                        "thumbnail": f"{BASE_URL}/static/{model_dir.name}/{metadata.get('thumbnail', 'thumbnail.jpg')}?t={cache_buster}",
                        "modelPath": f"{BASE_URL}/static/{model_dir.name}/{metadata.get('modelFile', 'model.glb')}?t={cache_buster}",
                        "transform": transform
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
    
    # Save 3D model file with sync to disk
    model_ext = Path(model.filename).suffix.lower()
    model_filename = f"model{model_ext}"
    model_path = model_dir / model_filename
    with open(model_path, "wb") as buffer:
        content = await model.read()
        buffer.write(content)
        buffer.flush()
        os.fsync(buffer.fileno())
    
    # Save thumbnail with sync to disk
    thumbnail_filename = "thumbnail.jpg"
    if thumbnail:
        thumbnail_ext = Path(thumbnail.filename).suffix.lower()
        thumbnail_filename = f"thumbnail{thumbnail_ext}"
        thumbnail_path = model_dir / thumbnail_filename
        with open(thumbnail_path, "wb") as buffer:
            content = await thumbnail.read()
            buffer.write(content)
            buffer.flush()
            os.fsync(buffer.fileno())
    
    # Add default identity transformation matrix to metadata
    metadata = {
        "id": model_id,
        "name": name,
        "description": description,
        "thumbnail": thumbnail_filename,
        "modelFile": model_filename,
        "createdAt": datetime.now().isoformat(),
        "lastUpdated": datetime.now().isoformat(),
        "transform": [
            [1, 0, 0, 0],  # Default identity matrix
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
    }
    
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    
    return {
        "success": True,
        "message": "Model created successfully",
        "modelId": model_id,
        "metadata": metadata
    }

@app.put("/api/models/{model_id}")
async def update_model(
    model_id: str,
    name: str = Form(...),
    description: str = Form(""),
    thumbnail: Optional[UploadFile] = File(None),
    model: Optional[UploadFile] = File(None)
):
    model_dir = MODELS_DIR / model_id
    
    if not model_dir.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        # Load existing metadata
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        metadata_updated = False
        model_filename = metadata.get("modelFile", "model.glb")
        thumbnail_filename = metadata.get("thumbnail", "thumbnail.jpg")

        # Update model file if provided
        if model and model.filename:
            # Remove old model file if exists
            old_model_path = model_dir / model_filename
            if old_model_path.exists():
                old_model_path.unlink()
            
            # Save new model file with sync
            model_ext = Path(model.filename).suffix.lower()
            model_filename = f"model{model_ext}"
            model_path = model_dir / model_filename
            with open(model_path, "wb") as buffer:
                content = await model.read()
                buffer.write(content)
                buffer.flush()
                os.fsync(buffer.fileno())
            
            metadata["modelFile"] = model_filename
            metadata_updated = True
        
        # Update thumbnail if provided
        if thumbnail and thumbnail.filename:
            # Remove old thumbnail if exists
            old_thumbnail_path = model_dir / thumbnail_filename
            if old_thumbnail_path.exists():
                old_thumbnail_path.unlink()
            
            # Save new thumbnail with sync
            thumbnail_ext = Path(thumbnail.filename).suffix.lower()
            thumbnail_filename = f"thumbnail{thumbnail_ext}"
            thumbnail_path = model_dir / thumbnail_filename
            with open(thumbnail_path, "wb") as buffer:
                content = await thumbnail.read()
                buffer.write(content)
                buffer.flush()
                os.fsync(buffer.fileno())
            
            metadata["thumbnail"] = thumbnail_filename
            metadata_updated = True
        
        # Update other metadata if changed
        if (metadata.get("name") != name or 
            metadata.get("description") != description):
            metadata["name"] = name
            metadata["description"] = description
            metadata_updated = True
        
        if metadata_updated:
            metadata["lastUpdated"] = datetime.now().isoformat()
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
        
        return {
            "success": True,
            "message": "Model updated successfully",
            "modelId": model_id,
            "metadata": metadata
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to update model: {str(e)}"
        )

@app.delete("/api/models/{model_id}")
async def delete_model(model_id: str):
    model_dir = MODELS_DIR / model_id
    
    if not model_dir.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
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

@app.post("/api/models/{model_id}/transform")
async def update_model_transform(
    model_id: str,
    transform: str = Form(..., description="JSON string of 4x4 transformation matrix")
):
    print("updating matrix")
    """
    Update only the transformation matrix of a specific model
    Format for transform:
    [
        [1, 0, 0, 0],  # Position/scale/rotation row 1
        [0, 1, 0, 0],  # Position/scale/rotation row 2
        [0, 0, 1, 0],  # Position/scale/rotation row 3
        [0, 0, 0, 1]   # Position/scale/rotation row 4
    ]
    """
    model_dir = MODELS_DIR / model_id
    
    if not model_dir.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        # Load existing metadata
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Parse and validate the transformation matrix
        try:
            transform_matrix = json.loads(transform)
            
            # Basic validation
            if not isinstance(transform_matrix, list) or \
               len(transform_matrix) != 4 or \
               any(not isinstance(row, list) or len(row) != 4 for row in transform_matrix):
                raise ValueError("Invalid matrix format")
                
            metadata["transform"] = transform_matrix
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid transform format: {str(e)}. Expected 4x4 matrix"
            )
        
        # Update last modified timestamp
        metadata["lastUpdated"] = datetime.now().isoformat()
        
        # Save updated metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        
        return {
            "success": True,
            "message": "Transform updated successfully",
            "modelId": model_id,
            "transform": transform_matrix
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to update transform: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)