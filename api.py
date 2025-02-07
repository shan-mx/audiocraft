from contextlib import asynccontextmanager
from time import time

from fastapi import FastAPI
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

from audiocraft.data.audio import audio_to_base64
from audiocraft.models import MAGNeT
from download import MODEL_NAME

model: MAGNeT


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = MAGNeT.get_pretrained(MODEL_NAME)
    model.set_generation_params(
        use_sampling=True,
        top_k=0,
        top_p=0.9,
        temperature=3.0,
        max_cfg_coef=10.0,
        min_cfg_coef=1.0,
        decoding_steps=[
            int(20 * model.lm.cfg.dataset.segment_duration // 10),
            10,
            10,
            10,
        ],
        span_arrangement="stride1",
    )
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenMusicRequest(BaseModel):
    text: str


@app.post("/gen")
def gen(request: GenMusicRequest):
    start_time = time()
    output = model.generate(descriptions=[request.text])[0]
    end_time = time()
    print(f"Finished generation in {end_time - start_time} seconds")
    return audio_to_base64(output.cpu(), model.sample_rate, "mp3")


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7867)
