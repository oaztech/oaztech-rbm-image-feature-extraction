from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from app.rest import rmb_rest

app = FastAPI()
app.include_router(rmb_rest.router, prefix="/machine-learning", tags=["MDL"])


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Restricted Boltzmann Machine",
        version="1.0.0",
        description="This API is a simple demo to creating features in facial recognition; specifically, it examines "
                    "extracting features from images using a Restricted Boltzmann Machine.",
        routes=app.routes,
        openapi_version="3.0.0"
    )
    openapi_schema["info"]["x-logo"] = {
        "url": "https://i1.hespress.com/wp-content/uploads/2021/09/schema_publisher_logo.jpg"
    }
    app.openapi_schema = openapi_schema

    return app.openapi_schema


app.openapi = custom_openapi
