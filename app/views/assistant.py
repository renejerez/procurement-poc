import time

import streamlit as st

import assistant
import lib
import rag

st.title("Assistente")
st.caption("Pergunte sobre os contratos e o conteúdo dos documentos. As respostas são geradas pelo Gemini a partir dos "
           "dados desta aplicação, com links para as fontes. Confirme sempre a informação nas fontes citadas.")

if not assistant.api_key():
    st.warning("Falta a chave da API do Gemini.")
    st.markdown("""
1. Crie uma chave gratuita em [Google AI Studio](https://aistudio.google.com/apikey).
2. **No Colab** (notebook 04): ícone da chave 🔑 → *Add new secret* → nome `GEMINI_API_KEY`, ative *Notebook access*,
   e volte a arrancar a aplicação.
3. **No Streamlit Cloud**: *Settings → Secrets* → `GEMINI_API_KEY = "..."`.
4. **Localmente**: variável de ambiente `GEMINI_API_KEY`.
""")
    st.stop()

path, _ = lib.db_snapshot()
indexed = rag.has_index(lib._connection(path).cursor())
if not indexed:
    st.info("O índice dos documentos ainda não existe (notebook 05_rag_index). O assistente responde só com os dados "
            "estruturados até lá.")

col1, col2 = st.columns([4, 1])
with col1:
    df = lib.contracts()
    options = [None] + df.sort_values("publication_date", ascending=False)["contract_id"].tolist()
    labels = dict(zip(df["contract_id"], df["object"].fillna("").str.slice(0, 90)))
    focus = st.selectbox("Contrato em foco (opcional)", options, key="focus",
                         format_func=lambda c: "Nenhum: perguntas sobre todos os contratos" if c is None else f"{c} · {labels.get(c, '')}")
with col2:
    st.write("")
    if st.button("Nova conversa", use_container_width=True):
        for k in ["chat", "chat_model", "messages", "trace"]:
            st.session_state.pop(k, None)
        st.rerun()

st.session_state.setdefault("messages", [])

EXAMPLES = [
    "Quais as 5 entidades que mais gastaram e quanto?",
    "Quantos ajustes diretos houve e qual o valor total?",
    "Que contratos têm alertas de proposta única? Explica o risco.",
    "Que prazos de pagamento aparecem nos cadernos de encargos?",
]
if focus:
    EXAMPLES = ["Resume este contrato.", "Que alertas tem este contrato e porquê?",
                "Quais são as penalidades previstas nos documentos?", "Qual é o prazo de execução e de garantia?"]

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("trace"):
            with st.expander("Como cheguei a esta resposta"):
                for t in msg["trace"]:
                    st.markdown(f"**{t['ferramenta']}** · {t['resultado']}")
                    st.json(t["argumentos"], expanded=False)

prompt = None
if not st.session_state["messages"]:
    st.markdown("**Exemplos:**")
    cols = st.columns(len(EXAMPLES))
    for col, ex in zip(cols, EXAMPLES):
        if col.button(ex, use_container_width=True):
            prompt = ex

prompt = st.chat_input("Escreva a sua pergunta...") or prompt
if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    message = prompt if not focus else f"[Contrato em foco: {focus}]\n{prompt}"
    st.session_state["trace"] = []
    with st.chat_message("assistant"):
        t0 = time.time()
        with st.status("A pensar...", expanded=True) as status:
            assistant.LIVE = status
            assistant._say("💬 Pergunta enviada ao Gemini")
            try:
                answer = assistant.send(message)
                status.update(label=f"Concluído em {time.time() - t0:.0f} s", state="complete", expanded=False)
            except Exception as e:
                answer = assistant.friendly_error(e)
                status.update(label="Erro", state="error", expanded=False)
            finally:
                assistant.LIVE = None
        st.markdown(answer)
        trace = st.session_state.get("trace", [])
        if trace:
            with st.expander("Como cheguei a esta resposta"):
                for t in trace:
                    st.markdown(f"**{t['ferramenta']}** · {t['resultado']}")
                    st.json(t["argumentos"], expanded=False)
    st.session_state["messages"].append({"role": "assistant", "content": answer, "trace": trace})
