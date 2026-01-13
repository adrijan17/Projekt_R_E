import os
from enum import Enum

# --- ENUMERACIJE SERVISA (GenieAI Standard) ---
class ServiceType(Enum):
    MICROSERVICE = 0
    MEGASERVICE = 1
    EMBEDDING = 1
    RETRIEVER = 2
    RERANK = 3
    LLM = 4

class ServiceRoleType(Enum):
    MICROSERVICE = 0
    MEGASERVICE = 1

# --- KONFIGURACIJSKE KONSTANTE (Iz GenieAI config.py) ---
# Modeli
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-3.5-turbo") # Default za studente (jeftiniji)
OPENAI_CHAT_TEMPERATURE = float(os.getenv("OPENAI_CHAT_TEMPERATURE", 0.01)) # Preciznost

# Retrieval postavke (ArangoDB stil)
RETRIEVER_SCORE_THRESHOLD = float(os.getenv("RETRIEVER_ARANGO_SCORE_THRESHOLD", 0.5))
RETRIEVER_MAX_RETURNED = int(os.getenv("RETRIEVER_ARANGO_TRAVERSAL_MAX_RETURNED", 3))

# System Prompts
SYSTEM_PROMPT_TEMPLATE = """
### Uloga:
Ti si stručan agronomski asistent za hrvatsko podneblje.
Koristi priložene "Rezultate pretrage" (iz lokalne baze DZS RH) kao primarni izvor istine.

### Rezultati pretrage (Kontekst): 
{context}

### Povijest:
{history}

### Pitanje:
{question}
"""
