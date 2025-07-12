from fastapi import FastAPI, Request, Form, BackgroundTasks
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from routes.api import router 
from app.auth import router as auth_router
#from models.book_model import reload_model
#from models.user_model import reload_user_model
from app.database import get_db

app = FastAPI()

"""
@app.on_event('startup')
async def startup_event():
    reload_model()
    reload_user_model()
"""

app.mount('/static', StaticFiles(directory='static'), name='static')

templates = Jinja2Templates(directory='templates')

app.include_router(router)
app.include_router(auth_router)

@app.get('/', response_class=HTMLResponse)
def root(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})


