from rag.rag import RAG
from typing import List, Annotated
from typing_extensions import TypedDict
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

from pymupdf import Document
import logging

logger = logging.getLogger("routes")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(levelname)s | %(name)s | %(message)s")
console_handler.setFormatter(formatter)

# Attach the handler to the logger
logger.addHandler(console_handler)

# Disable propagation if you don't want library logs
logger.propagate = False


rag_router = APIRouter()


rag = RAG()


class RagResponse(TypedDict):
    """
    RAG response template.
    """

    success: bool
    message: str
    contexts: List[str] | None


class QueryParams(BaseModel):
    query: str
    """User query"""

    message_history: Optional[List[str]] = None
    """List of previous messages in given conversation"""


@rag_router.post("/query/{conversation_id}")
async def query(conversation_id: int, params: QueryParams) -> JSONResponse:
    """
    Query the RAG system.
    """

    query = params.query
    history = params.message_history if params.message_history else []

    logger.debug(f"/query/ Received message: {query} with history size: {len(history)}")

    response: RagResponse = {"success": True, "message": "", "contexts": None}

    # TODO :: Error handling when rag will be ready

    try:
        rag_response = rag.process_query(query, conversation_id)

        response["message"] = rag_response
        # TODO :: Add contexts
        response["contexts"] = ["Contexts not implemented yet"]

        return JSONResponse(content=response)

    except Exception as e:
        logging.error(f"Error during generation of RAG response. {e}")

        response["message"] = "Internal server error when processing query"
        response["success"] = False

        return JSONResponse(content=response, status_code=500)


class UploadResponse(TypedDict):
    success: bool
    message: str | None


@rag_router.post("/upload/{conversation_id}")
async def upload_documents(
    conversation_id: int, files: Annotated[List[UploadFile], File(...)] = []
) -> JSONResponse:
    """
    Upload documents to RAG system.
    """
    logger.info(
        f"Received {len(files)} files for conversation {conversation_id}.\nFile data {files}"
    )

    response: UploadResponse = {"success": True, "message": "Files uploaded"}

    if not files:
        logger.debug("No files provided")
        response["success"] = False
        response["message"] = "No files provided"

        return JSONResponse(content=response, status_code=400)

    for file in files:
        logger.info(
            f"Processing file: {file.filename} for conversation {conversation_id}"
        )
        files_bytes = await file.read()
        doc = Document(stream=files_bytes, filetype="pdf")
        rag.process_document(doc, conversation_id)

    return JSONResponse(content=response, status_code=201)


class DeleteResponse(TypedDict):
    status: bool
    message: str


@rag_router.delete("/delete/{converastion_id}")
async def delete_conversation(converastion_id: int) -> DeleteResponse:
    """
    Delete all conversation data.
    """
    logger.debug(f"Deleting conversation {converastion_id} data")

    return {"status": True, "message": "Collection deleted"}
