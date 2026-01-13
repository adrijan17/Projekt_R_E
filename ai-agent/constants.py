from enum import Enum

class ServiceType(Enum):
    """
    Definira tipove servisa unutar našeg AI Agenta.
    Preuzeto iz GenieAI arhitekture.
    """
    EMBEDDING = 1    # Prevaranje teksta u vektore (opcionalno za tebe)
    RETRIEVER = 2    # Pretraživanje baze znanja (tvoj JSON pretraživač)
    RERANK = 3       # Sortiranje rezultata po relevantnosti
    LLM = 4          # OpenAI / GenieAI model
    TRANSLATOR = 24  # Prevođenje (ako zatreba)
    
class ServiceRoleType(Enum):
    MICROSERVICE = 0
    MEGASERVICE = 1  # Tvoj glavni agent je 'Megaservice' koji upravlja ostalima
