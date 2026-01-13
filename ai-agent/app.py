import streamlit as st
from agronomy_agent import AgronomistAgent

# --- KONFIGURACIJA STRANICE ---
st.set_page_config(page_title="AgroGenie AI", page_icon="🌱", layout="centered")

# Naslov i opis
st.title("🌱 AgroGenie: Pametni Agronom")
st.markdown("""
Ovaj AI agent koristi **RAG (Retrieval-Augmented Generation)** tehnologiju i službene podatke DZS-a 
kako bi vam dao precizne savjete o uzgoju kultura u Hrvatskoj.
""")

# Inicijalizacija agenta (samo jednom)
if "agent" not in st.session_state:
    st.session_state.agent = AgronomistAgent()

# Inicijalizacija povijesti chata
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- PRIKAZ POVIJESTI CHATA ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- UNOS KORISNIKA ---
if prompt := st.chat_input("Pitajte o kukuruzu, pšenici, lozi..."):
    # 1. Prikaži korisnikovu poruku odmah
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Agent razmišlja (spinner)
    with st.chat_message("assistant"):
        with st.spinner("Analiziram bazu znanja i DZS statistiku..."):
            response = st.session_state.agent.process_request(prompt)
            st.markdown(response)
    
    # 3. Spremi odgovor
    st.session_state.messages.append({"role": "assistant", "content": response})

# --- SIDEBAR (Opcije za demonstraciju) ---
with st.sidebar:
    st.header("🔧 Debug Panel")
    if st.checkbox("Prikaži sirove podatke"):
        st.json(st.session_state.agent.knowledge_base)
    
    st.info("Sustav koristi GraphRAG logiku za filtriranje po kategorijama (Voće, Žitarice...).")
