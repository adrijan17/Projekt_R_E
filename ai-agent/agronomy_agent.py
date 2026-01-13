import os
import json
import re  # Za obradu teksta
from datetime import datetime

# Pokušaj importa OpenAI biblioteke
try:
    from openai import OpenAI
except ImportError:
    print("MOLIM INSTALIRAJTE: pip install openai")
    exit(1)

# --- KONFIGURACIJA (Kao u GenieAI configs) ---
# Zamijeni ovo pravim API ključem ili postavi environment varijablu
API_KEY = os.getenv("OPENAI_API_KEY", "sk-proj-TVOJ-API-KLJUC-OVDJE")

# Postavke iz 'ChatCompletionRequest' definicije
CONF = {
    "model": "gpt-3.5-turbo", # Ili gpt-4o
    "temperature": 0.01,       # Jako nisko (kako je definirano u GenieAI protocolu)
    "max_tokens": 1024,
    "top_p": 0.95,
    "frequency_penalty": 0.0,
    "score_threshold": 0.1     # Minimalna relevantnost dokumenta
}

# Inicijalizacija klijenta
client = OpenAI(api_key=API_KEY)

# --- KLASA AGENTA ---
class AgronomistAgent:
    def __init__(self, knowledge_base_path="hr_crops.json"):
        self.kb_path = knowledge_base_path
        self.knowledge_base = self._load_data()
        self.chat_history = [] 
        
        # PROMPT Template preuzet iz 'genieai_chatqna.py' i preveden
        self.system_prompt_template = """
### Uloga:
Ti si koristan, stručan i iskren agronoski asistent. Tvoj cilj je pomoći korisniku s pitanjima o poljoprivredi.
Molim te, koristi priložene "Rezultate pretrage" (iz lokalne baze znanja DZS RH) da odgovoriš na pitanje.

Pravila:
1. Ako odgovor nije u rezultatima pretrage, iskoristi svoje opće znanje ali to JASNO NAGLASI.
2. Nemoj izmišljati lažne informacije.
3. Odgovor mora biti na hrvatskom jeziku.

### Rezultati pretrage (Kontekst): 
{context}

### Tvoj zadatak:
Odgovori na zadnje pitanje korisnika na temelju gornjeg konteksta i povijesti razgovora.
"""

    def _load_data(self):
        """Učitava JSON bazu (Simulacija ArangoDB-a)"""
        if not os.path.exists(self.kb_path):
            print(f"Upozorenje: Baza {self.kb_path} nije pronađena. Koristim praznu.")
            return []
        try:
            with open(self.kb_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"[INFO] Učitano {len(data)} zapisa o kulturama.")
                return data
        except Exception as e:
            print(f"[ERROR] Greška kod učitavanja JSON-a: {e}")
            return []

    def _simple_retriever(self, query):
        """
        Pojednostavljeni 'Retriever' servis.
        Traži ključne riječi iz Queryja u JSON bazi.
        """
        hits = []
        query_terms = query.lower().split()
        
        for item in self.knowledge_base:
            # Spoji sva polja u jedan tekst radi pretrage
            item_text = json.dumps(item, ensure_ascii=False).lower()
            
            # Bodovanje (Score)
            score = 0
            for term in query_terms:
                if len(term) > 3 and term in item_text:
                    score += 0.2
            
            # Ako je naslov kulture direktno u pitanju, veliki bonus
            if item.get("kultura", "").lower() in query.lower():
                score += 1.0

            if score >= CONF["score_threshold"]:
                hits.append({"doc": item, "score": score})
        
        # Sortiraj po relevantnosti (Simulacija 'Rerank' servisa)
        hits.sort(key=lambda x: x["score"], reverse=True)
        
        # Vrati top 3 rezultata kao string
        if not hits:
            return "Nema relevantnih podataka u lokalnoj bazi."
            
        context_str = ""
        for hit in hits[:3]:
            doc = hit['doc']
            context_str += f"- KULTURA: {doc.get('kultura')}\n"
            context_str += f"  PREPORUKA REGIJE: {doc.get('regija_preporuka')}\n"
            context_str += f"  STATISTIKA PRNOSA: {doc.get('dzs_statistika')}\n"
            context_str += f"  UVJETI UZGOJA: {doc.get('uvjeti_uzgoja')}\n"
            context_str += "---\n"
            
        return context_str

    def _format_history(self, max_chars=2000):
        """
        Simulira 'align_inputs' logiku za rezanje povijesti.
        Ne šaljemo sve jer je skupo i nepotrebno.
        """
        history_str = ""
        total_chars = 0
        
        # Idemo od zadnje poruke prema prvoj
        for msg in reversed(self.chat_history):
            msg_str = f"{msg['role']}: {msg['content']}\n"
            if total_chars + len(msg_str) > max_chars:
                break
            history_str = msg_str + history_str
            total_chars += len(msg_str)
            
        return history_str

    def generate_response(self, user_question):
        # 1. RETRIEVAL (Dohvati podatke)
        context = self._simple_retriever(user_question)
        
        # 2. PROMPT CONSTRUCTION
        system_msg = self.system_prompt_template.format(context=context)
        history_str = self._format_history()
        
        full_msg_list = [
            {"role": "system", "content": system_msg},
            # Ovdje možemo ubaciti i sažetu povijest ako model podržava dugačak kontekst
            # Za jednostavnost, samo dodajemo zadnji user query u API poziv dolje
        ]
        
        # Dodaj povijest u messages listu
        for msg in self.chat_history[-4:]: # Samo zadnja 2 kruga razgovora
            full_msg_list.append(msg)
            
        full_msg_list.append({"role": "user", "content": user_question})

        # 3. LLM CALL (OpenAI API koji oponaša GenieAI servis)
        try:
            print(f"[AI] Razmišljam na temelju {len(context)} znakova konteksta...")
            response = client.chat.completions.create(
                model=CONF["model"],
                messages=full_msg_list,
                temperature=CONF["temperature"],
                max_tokens=CONF["max_tokens"],
                frequency_penalty=CONF["frequency_penalty"]
            )
            
            answer = response.choices[0].message.content
            
            # Ažuriraj povijest
            self.chat_history.append({"role": "user", "content": user_question})
            self.chat_history.append({"role": "assistant", "content": answer})
            
            return answer

        except Exception as e:
            return f"Greška u komunikaciji s AI servisom: {str(e)}"

# --- DEMO NAČIN RADA ---
if __name__ == "__main__":
    # Testiranje
    agent = AgronomistAgent()
    
    print("\n=== GENIE.AI AGRONOMIST AGENT (Lokalna Verzija) ===")
    print("Upišite 'q' za izlaz.\n")
    
    while True:
        pitanje = input("TI: ")
        if pitanje.lower() in ['q', 'exit']:
            break
            
        odgovor = agent.generate_response(pitanje)
        print(f"AGENT: {odgovor}\n")
