"""
Root execution script for Synk Corp Support Bot.
Run with:
    python main.py
Or with uvicorn:
    uvicorn main:app --reload --port 8000
"""

from app.main import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
