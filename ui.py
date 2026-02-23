import streamlit as st
from main_orchestrator import run_hub

# 1. PAGE CONFIG
st.set_page_config(page_title="Corporate AI Hub", page_icon="🏢")
st.title("🏢 Corporate Consultancy Hub")
st.markdown("---")

# 2. INITIALIZE CHAT HISTORY
if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. DISPLAY CHAT MESSAGES
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. CHAT INPUT
if prompt := st.chat_input("How can I help you today?"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 5. GENERATE AI RESPONSE
    with st.chat_message("assistant"):
        with st.spinner("Consulting specialists..."):
            # Call your orchestrator!
            response = run_hub(prompt)
            st.markdown(response)
    
    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": response})
    # with st.sidebar:
    # st.header("Department Specialists")
    # st.success("✅ Legal (Lead)")
    # st.warning("⏳ HR (In Progress)")
    # st.warning("⏳ IT (In Progress)")
    # st.warning("⏳ Customer Success (In Progress)")
