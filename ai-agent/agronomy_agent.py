import os
import json
from datetime import datetime
from constants import (
    ServiceType, OPENAI_CHAT_MODEL, OPENAI_CHAT_TEMPERATURE,
    RETRIEVER_SCORE_THRESHOLD, RETRIEVER_MAX_RETURNED, SYSTEM_PROMPT_TEMPLATE
)

try:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except ImportError:
    print("MOLIM INSTALIRAJTE: pip install openai")
    exit(1)

class AgronomistAgent:
    def __init__(self, knowledge_base_path="hr_crops.json"):
        self.kb_path = knowledge_base_path
        self.knowledge_base = self._load_data()
        self.chat_history = []

    def _log(self, service: ServiceType, message):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [{service.name}] {message}")

    def _load_data(self):
        if not os.path.exists(self.kb_path): return []
        with open(self.kb_path, 'r', encoding='utf-8') as f: return json.load(f)

    # --- SIMULACIJA ARANGO FILTRIRANJA (Inspirirano linijama 700-730 tvoje datoteke) ---
    def _apply_filters(self, items, category_filter=None):
        if not category_filter:
            return items
        
        filtered = [i for i in items if i.get("kategorija", "").lower() == category_filter.lower()]
        self._log(ServiceType.RETRIEVER, f"Filtriranje po kategoriji '{category_filter}': {len(items)} -> {len(filtered)}")
        return filtered

    def _retriever_service(self, query, category_filter=None):
        self._log(ServiceType.RETRIEVER, f"Graph Traversal Start: '{query}'")
        
        # 1. Prvo filtriraj (Simulacija AQL FILTER klauzule)
        candidates = self._apply_filters(self.knowledge_base, category_filter)
        
        hits = []
        query_terms = query.lower().split()
        
        for item in candidates:
            # 2. Vektorska sličnost (Simulirano preko ključnih riječi)
            item_text = json.dumps(item, ensure_ascii=False).lower()
            score = 0
            
            for term in query_terms:
                if len(term) > 3 and term in item_text:
                    score += 0.2
            if item.get("kultura", "").lower() in query.lower():
                score += 1.0 # Jaki signal (Exact match)

            # Simulacija 'fetch_neighborhoods' - dohvaćamo sve ako je score visok
            if score >= RETRIEVER_SCORE_THRESHOLD:
                hits.append({"doc": item, "score": score})
        
        return hits

    def process_request(self, user_question):
        # DETEKCIJA NAMJERE (Jednostavna logika)
        cat_filter = None
        if "voće" in user_question.lower() or "voćk" in user_question.lower():
            cat_filter = "Voće"
        elif "žit" in user_question.lower():
            cat_filter = "Žitarice"

        # 1. RETRIEVE
        raw_hits = self._retriever_service(user_question, category_filter=cat_filter)
        
        # 2. RERANK
        raw_hits.sort(key=lambda x: x["score"], reverse=True)
        best_docs = [h['doc'] for h in raw_hits[:RETRIEVER_MAX_RETURNED]]
        
        # Formatiranje konteksta
        context_str = ""
        for doc in best_docs:
            context_str += f"- KULTURA: {doc.get('kultura')} ({doc.get('kategorija')})\n"
            context_str += f"  UVJETI: {doc.get('uvjeti_uzgoja')}\n"
            # Ovdje bi GraphRAG dodao "RELATED INFORMATION"
            context_str += "---\n"

        history_str = "\n".join([f"{m['role']}: {m['content']}" for m in self.chat_history[-2:]])

        # 3. LLM GENERATION
        final_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            context=context_str if context_str else "Nema podataka.",
            history=history_str,
            question=user_question
        )
        
        self._log(ServiceType.LLM, "Generiranje odgovora...")
        try:
            response = client.chat.completions.create(
                model=OPENAI_CHAT_MODEL,
                messages=[{"role": "user", "content": final_prompt}],
                temperature=OPENAI_CHAT_TEMPERATURE
            )
            answer = response.choices[0].message.content
        except Exception as e:
            answer = f"Greška: {e}"
        
        self.chat_history.append({"role": "user", "content": user_question})
        self.chat_history.append({"role": "assistant", "content": answer})
        return answer

if __name__ == "__main__":
    agent = AgronomistAgent()
    print("--- AGRONOMY AGENT POKRENUT ---")
    while True:
        q = input("\nPitanje: ")
        if q.lower() == 'q': break
        print(f"\n>> {agent.process_request(q)}")
