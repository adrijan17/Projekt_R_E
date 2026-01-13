import os
import json
from datetime import datetime
try:
    from openai import OpenAI
except ImportError:
    print("MOLIM INSTALIRAJTE: pip install openai")
    exit(1)

# Uvozimo konstante iz druge datoteke
from constants import ServiceType, ServiceRoleType

# --- KONFIGURACIJA ---
API_KEY = os.getenv("OPENAI_API_KEY", "tvoj-api-kljuc-s-openai")

CONF = {
    "model": "gpt-3.5-turbo",
    "temperature": 0.01,
    "max_tokens": 1024,
    "score_threshold": 0.2
}

client = OpenAI(api_key=API_KEY)

class AgronomistAgent:
    def __init__(self, knowledge_base_path="hr_crops.json"):
        self.kb_path = knowledge_base_path
        self.knowledge_base = self._load_data()
        self.chat_history = []
        self.role = ServiceRoleType.MEGASERVICE
        
        # Inicijalizacija prompta (iz prijašnjih koraka)
        self.system_prompt_template = """
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

    def _log(self, service: ServiceType, message):
        """Pomoćna funkcija za ispis u stilu mikroservisa"""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [{service.name}] {message}")

    def _load_data(self):
        if not os.path.exists(self.kb_path):
            return []
        with open(self.kb_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _retriever_service(self, query):
        """Simulira RETRIEVER mikroservis"""
        self._log(ServiceType.RETRIEVER, f"Tražim podatke za: '{query}'")
        
        hits = []
        query_terms = query.lower().split()
        
        for item in self.knowledge_base:
            item_text = json.dumps(item, ensure_ascii=False).lower()
            score = 0
            
            # Jednostavna logika bodovanja
            for term in query_terms:
                if len(term) > 3 and term in item_text:
                    score += 0.2
            if item.get("kultura", "").lower() in query.lower():
                score += 1.0

            if score >= CONF["score_threshold"]:
                hits.append({"doc": item, "score": score})
        
        self._log(ServiceType.RETRIEVER, f"Pronađeno {len(hits)} relevantnih dokumenata.")
        return hits

    def _rerank_service(self, hits):
        """Simulira RERANK mikroservis"""
        if not hits:
            return []
        
        # Sortiranje po score-u
        hits.sort(key=lambda x: x["score"], reverse=True)
        top_hits = hits[:3] # Uzmi top 3
        
        self._log(ServiceType.RERANK, f"Odabrano top {len(top_hits)} rezultata.")
        return [h['doc'] for h in top_hits]

    def _llm_service(self, prompt):
        """Simulira LLM mikroservis"""
        self._log(ServiceType.LLM, "Šaljem upit AI modelu...")
        try:
            response = client.chat.completions.create(
                model=CONF["model"],
                messages=[{"role": "user", "content": prompt}], # Pojednostavljeno
                temperature=CONF["temperature"]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Greška: {e}"

    def process_request(self, user_question):
        """
        Orkestracija servisa (Kao MegaService u GenieAI)
        Flow: Retriever -> Rerank -> LLM
        """
        # 1. RETRIEVE
        raw_hits = self._retriever_service(user_question)
        
        # 2. RERANK
        best_docs = self._rerank_service(raw_hits)
        
        # Formatiranje konteksta za LLM
        context_str = ""
        if not best_docs:
            context_str = "Nema specifičnih podataka u bazi."
        else:
            for doc in best_docs:
                context_str += f"- {json.dumps(doc, ensure_ascii=False)}\n"

        # Formatiranje povijesti
        history_str = "\n".join([f"{m['role']}: {m['content']}" for m in self.chat_history[-2:]])

        # 3. LLM GENERATION
        final_prompt = self.system_prompt_template.format(
            context=context_str,
            history=history_str,
            question=user_question
        )
        
        answer = self._llm_service(final_prompt)
        
        # Update history
        self.chat_history.append({"role": "user", "content": user_question})
        self.chat_history.append({"role": "assistant", "content": answer})
        
        return answer

if __name__ == "__main__":
    agent = AgronomistAgent()
    print("--- GENIE.AI AGRONOM COPY STARTED ---")
    
    while True:
        q = input("\nPitanje (q za kraj): ")
        if q == 'q': break
        print(f"\nODGOVOR: {agent.process_request(q)}")
