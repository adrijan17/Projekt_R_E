import streamlit as st
from ai_agent import AgronomistAgent

st.set_page_config(page_title="AgroGenie AI", page_icon="🌱", layout="centered")

st.title("🌱 AgroGenie: Pametni Agronom")
st.markdown("""
Ovaj AI agent koristi službene podatke DZS-a kako bi Vam dao precizne savjete o uzgoju kultura u Hrvatskoj.
""")

# Inicijalizacija agenta
if "agent" not in st.session_state:
    st.session_state.agent = AgronomistAgent()

# Inicijalizacija povijesti chata
if "messages" not in st.session_state:
    st.session_state.messages = []

# Prikaz povijesti chata
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Za unos korisnika
if prompt := st.chat_input("Pitajte o kukuruzu, pšenici, lozi..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Analiziram bazu znanja i DZS statistiku..."):
            response = st.session_state.agent.process_request(prompt)
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})

with st.sidebar:
    st.header("🔧 Debug Panel")
    if st.checkbox("Prikaži sirove podatke"):
        st.json(st.session_state.agent.knowledge_base)
    
    st.info("Sustav koristi GraphRAG logiku za filtriranje po kategorijama (Voće, Žitarice...).")
