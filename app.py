"""
app.py
Entry point for ASGI/WSGI Serverless deployment (Vercel, AWS Lambda, Render).
Exports top-level 'app' instance.
"""

from api.index import app

# Top-level ASGI handler exported for Vercel
__all__ = ["app"]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
