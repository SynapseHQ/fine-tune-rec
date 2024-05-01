from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from utils.get_user_feed import get_user_feed, get_user_recommendations

app = FastAPI()


templates = Jinja2Templates(directory="templates")

@app.get("/")
async def root(request: Request, response_class = HTMLResponse):
    return templates.TemplateResponse(
     name="index.html",
     context={"request": request}
    )


@app.get("/user/")
async def user_feed(request: Request, response_class = HTMLResponse):
    posts = get_user_feed(user_id=request.query_params['user_id'])
    return templates.TemplateResponse(
     name="feed.html",
     context={"posts": posts, "request": request, "user_id": request.query_params['user_id']}
    )


@app.get("/recommendations/")
async def user_recommendations(request: Request, response_class = HTMLResponse):
    posts = get_user_recommendations(prompt=request.query_params['prompt'], user_id=request.query_params['user_id'])
    return templates.TemplateResponse(
     name="recommendations.html",
     context={"posts": posts, "request": request, "user_id": request.query_params['user_id']}
    )
