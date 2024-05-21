from fastapi import FastAPI
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

from audiocraft.data.audio import audio_to_base64
from audiocraft.models import MAGNeT

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenMusicRequest(BaseModel):
    text: str


model = MAGNeT.get_pretrained("facebook/magnet-small-30secs")

model.set_generation_params(
    use_sampling=True,
    top_k=0,
    top_p=0.9,
    temperature=3.0,
    max_cfg_coef=10.0,
    min_cfg_coef=1.0,
    decoding_steps=[int(20 * model.lm.cfg.dataset.segment_duration // 10), 10, 10, 10],
    span_arrangement="stride1",
)


@app.post("/gen")
async def gen(request: GenMusicRequest):
    output = model.generate(descriptions=[request.text])[0]
    return audio_to_base64(output.cpu(), model.sample_rate, "ogg")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7867)
