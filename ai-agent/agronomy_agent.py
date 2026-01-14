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
    print("GREŠKA: Nedostaje biblioteka. Instaliraj: pip install openai")
    exit(1)

class AgronomistAgent:
    def __init__(self, knowledge_base_path="usjevi.json"):
        self.kb_path = knowledge_base_path
        self.knowledge_base = self._load_data()
        self.chat_history = []

    def _log(self, service: ServiceType, message):
        """Pomoćna funkcija za ispis logova u konzoli"""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [{service.name}] {message}")

    def _load_data(self):
        """Učitava JSON bazu"""
        if not os.path.exists(self.kb_path): 
            print(f"Upozorenje: Datoteka {self.kb_path} ne postoji!")
            return []
        with open(self.kb_path, 'r', encoding='utf-8') as f: 
            data = json.load(f)
            print(f"[SYSTEM] Učitano {len(data)} kultura iz baze.")
            return data

    def _apply_filters(self, items, category_filter=None):
        """Simulira GraphRAG filtriranje (ArangoDB logika)"""
        if not category_filter:
            return items
        
        filtered = [i for i in items if i.get("kategorija", "").lower() == category_filter.lower()]
        self._log(ServiceType.RETRIEVER, f"Filter '{category_filter}': {len(items)} -> {len(filtered)} dokumenata")
        return filtered

    def _format_statistics(self, stats):
        """Pretvara složeni JSON objekt statistike u čitljiv tekst za AI"""
        if isinstance(stats, dict):
            lines = []
            for k, v in stats.items():
                clean_key = k.replace("_", " ").capitalize()
                lines.append(f"{clean_key}: {v}")
            return "; ".join(lines)
        return str(stats)

    def _retriever_service(self, query, category_filter=None):
        """Glavna logika pretrage"""
        self._log(ServiceType.RETRIEVER, f"Tražim: '{query}'")
        
        # Filtriranje
        candidates = self._apply_filters(self.knowledge_base, category_filter)
        
        hits = []
        query_terms = query.lower().split()
        
        for item in candidates:
            # Simulirana vektorska sličnost
            # Pretvaranje cijelog item-a u tekst da možemo tražiti ključne riječi
            item_text = json.dumps(item, ensure_ascii=False).lower()
            score = 0
            
            for term in query_terms:
                if len(term) > 3 and term in item_text:
                    score += 0.2
            
            # Ako se ime kulture spominje u pitanju
            if item.get("kultura", "").lower() in query.lower():
                score += 2.0 

            if score >= RETRIEVER_SCORE_THRESHOLD:
                hits.append({"doc": item, "score": score})
        
        return hits

    def process_request(self, user_question):
        # Detekcija kategorije
        cat_filter = None
        q_lower = user_question.lower()
        if any(x in q_lower for x in ["voće", "voćk", "jabuk", "krušk", "maslin", "grožđ", "mandarin"]):
            cat_filter = "Voće"
        elif any(x in q_lower for x in ["žit", "pšenic", "kukuruz", "ječam"]):
            cat_filter = "Žitarice"
        elif any(x in q_lower for x in ["povr", "rajčic", "krastav", "lubenic"]):
            cat_filter = "Povrće"

        # Retrieve
        raw_hits = self._retriever_service(user_question, category_filter=cat_filter)
        
        # Rerank
        raw_hits.sort(key=lambda x: x["score"], reverse=True)
        best_docs = [h['doc'] for h in raw_hits[:RETRIEVER_MAX_RETURNED]]
        
        # Formatiranje konteksta za AI
        context_str = ""
        if not best_docs:
            context_str = "Nema specifičnih podataka u DZS bazi za ovaj upit."
        else:
            for doc in best_docs:
                stats_readable = self._format_statistics(doc.get('dzs_statistika', {}))
                
                context_str += f"""
--- PREDMET: {doc.get('kultura')} ({doc.get('kategorija')})
 REGIJE: {', '.join(doc.get('regija_preporuka', []))}
 STATITISTIKA (DZS 2024): {stats_readable}
 UVJETI UZGOJA: {doc.get('uvjeti_uzgoja')}
"""

        history_str = "\n".join([f"{m['role']}: {m['content']}" for m in self.chat_history[-2:]])

        # LLM generiranje
        final_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            context=context_str,
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
            answer = f"Greška u komunikaciji s AI servisom: {e}\n(Provjeri API ključ)"
        
        self.chat_history.append({"role": "user", "content": user_question})
        self.chat_history.append({"role": "assistant", "content": answer})
        
        return answer


if __name__ == "__main__":
    agent = AgronomistAgent()
    print("\n🌱 AgroGenie (DZS Edition) je spreman!")
    print("Probajte pitati: 'Kakvo je stanje s mandarinama?' ili 'Koliko pšenice je proizvedeno?'\n")
    
    while True:
        q = input("Pitanje (q za izlaz): ")
        if q.lower() in ['q', 'exit']: break
        resp = agent.process_request(q)
        print(f"\nAGENT: {resp}\n")
        print("-" * 50)
