import io

import numpy as np
from PIL import Image
from fastapi import APIRouter, File

from app.dto.Item_dto import Item
from app.service import rbm_service

router = APIRouter()


@router.post("/extract-feature-image", response_model=None)
async def extractFeatureImage(file: bytes = File(description="A file read as bytes")):
    image = Image.open(io.BytesIO(file))
    img_arr = np.asarray(image)
    individual = rbm_service.extractFeatureImage(img_arr)

    return int(individual)


@router.post("/training-modal", response_model=None)
async def training_modal():
    rbm_service.training_modal()


@router.post("/items/", response_model=Item)
async def create_item(item: Item):
    return item
