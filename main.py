from dotenv import load_dotenv
import os
from uuid import UUID
from tqdm import tqdm
from datasets import load_dataset
import torch

from src.rag import ClassicRAG
from src.llm_client import BielikLLM, OpenAILLM
from src.document import DocumentLoaderFactory



load_dotenv()

bielik = BielikLLM(
    api_url=os.getenv("PG_API_URL") or "",
    username=os.getenv("PG_API_USERNAME") or "",
    password=os.getenv("PG_API_PASSWORD") or "",
)

chatgpt = OpenAILLM()
rag = ClassicRAG(bielik)

#
# def load_documents_to_chatbot(folder_path: str, chatbot: ClassicRAG,
#                               conversation_id: UUID):
#     documents_batch = []
#     total = 0
#
#     for filename in os.listdir(folder_path):
#         file_path = os.path.join(folder_path, filename)
#
#         if filename.endswith((".txt", ".md")) and os.path.isfile(file_path):
#             document = DocumentLoaderFactory.load(file_path)
#             total += 1
#             rag.process_document(document, conversation_id)
#             print(f"Processed {total} documents from {len(os.listdir(folder_path))}.")


# load_documents_to_chatbot(
#     folder_path="data",
#     chatbot=rag,
#     conversation_id=0
# )

# torch.cuda.empty_cache()
response = rag.process_query("Napisz 1:1 wymienione zrodla", 0)
print(response)

# rag.client.remove_collection(0)

dataset = load_dataset()