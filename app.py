from fastapi import FastAPI, Request
from slowapi.errors import RateLimitExceeded
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from pydantic import BaseModel
import subprocess

app = FastAPI()

# Define Rate limiter and add exception handler
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Define the response model for your API endpoint
class PotatoImage(BaseModel):
    image: bytes
    content_type: str

# Define the endpoint to generate the image
@app.get("/potato", response_model=PotatoImage)
@limiter.limit("100/minute")
async def generate_potato(request: Request):
    
    # Run the script to generate the image and capture the output
    output = subprocess.check_output(["python", "gen_potato.py"])

    # Return the image as a bytes object with content type "image/png"
    return PotatoImage(image=output, content_type="image/png")