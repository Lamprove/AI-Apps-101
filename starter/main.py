from fastapi import FastAPI, HTTPException, Request    
from fastapi.responses import HTMLResponse, FileResponse   
from fastapi.templating import Jinja2Templates    
from pydantic import BaseModel
from typing import Union   
import os
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
import uvicorn
import requests 

app = FastAPI()    

favicon_path = 'favicon.ico'   
templates = Jinja2Templates(directory="templates") 
API_URL = f'{os.getenv("LAMPROVE_API_URL")}/flan-t5-base/v1'
API_KEY = os.getenv('LAMPROVE_API_KEY')

class InputData(BaseModel):    
    text: str 

@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    return FileResponse(favicon_path) 
 
@app.get("/", response_class=HTMLResponse)    
def read_root(request: Request):    
    return templates.TemplateResponse("index.html", {"request": request})  
  
@app.get("/starterai", response_class=HTMLResponse) 
def read_evaluate(request: Request):  
    return templates.TemplateResponse("starterai.html", {"request": request})  

@app.post("/summarise_text")    
def summarise_text(data: InputData):    
    if data.text: 
        headers = {
                'accept': 'application/json',
                'Content-Type': 'application/json',
                'Ocp-Apim-Subscription-Key': API_KEY
            }
        text = BeautifulSoup(data.text, 'html.parser').text
        min_length = int(0.2 * len(word_tokenize(text)))
        if len(text.split()) < 10:
            raise HTTPException(status_code=404, detail="Text too short")
        gen_args = {
            "do_sample": False,
            "top_p": 1.0,
            "min_length": min_length,
            "max_new_tokens": 1000,
            "repetition_penalty": 2.0
        }
        instruction = 'summarize: '
        json_data = {
            'text': text,
            'instruction': instruction,
            'gen_args': gen_args
        }
        try:
            response = requests.post(f'{API_URL}/get_model_response', headers=headers, json=json_data)
            response = response.json()['response']
        except:
            response = 'Error with service. Please try again.'
        response

        return {"result": response}    
    else:    
        raise HTTPException(status_code=400, detail="Text not provided")   
    
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))