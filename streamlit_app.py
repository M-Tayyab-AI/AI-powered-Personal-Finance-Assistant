import io
from contextlib import redirect_stdout

import streamlit as st

from main import chat, clear_history
from schema import ChatRequest


APP_TITLE = "AI-Powered Personal Financial Assistant"
SUPPORTED_USER_IDS = [1, 2, 3]


def _init_state() -> None:
    if "ui_messages" not in st.session_state:
        st.session_state.ui_messages = {uid: [] for uid in SUPPORTED_USER_IDS}


def _ensure_user_store(user_id: int) -> None:
    if user_id not in st.session_state.ui_messages:
        st.session_state.ui_messages[user_id] = []


def _ask_assistant(user_id: int, query: str) -> tuple[str, str]:
    logs_buffer = io.StringIO()
    try:
        with redirect_stdout(logs_buffer):
            response = chat(ChatRequest(user_id=user_id, query=query))
        return response.response, logs_buffer.getvalue().strip()
    except Exception as exc:  # pragma: no cover
        return (
            "I ran into an error while processing your request. Please try again.",
            f"Error: {exc}\n\n{logs_buffer.getvalue().strip()}".strip(),
        )


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="💰",
        layout="wide",
    )

    _init_state()

    st.title(APP_TITLE)
    st.caption(
        "Ask finance-related questions about your spending, budgeting, and savings."
    )

    header_col_1, header_col_2 = st.columns([3, 1])
    with header_col_1:
        selected_user_id = st.selectbox(
            "Select User ID",
            options=SUPPORTED_USER_IDS,
            index=0,
            help="Default is user 1. You can switch to user 2 or 3.",
        )
    with header_col_2:
        if st.button("Clear Current User Chat", use_container_width=True):
            clear_history(selected_user_id)
            st.session_state.ui_messages[selected_user_id] = []
            st.success(f"Cleared conversation for user {selected_user_id}.")

    _ensure_user_store(selected_user_id)
    current_messages = st.session_state.ui_messages[selected_user_id]

    st.divider()

    if not current_messages:
        st.info(
            "Start by asking something like: "
            "'How much did I spend last week?' or 'Give me a budgeting strategy.'"
        )

    for msg in current_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("logs"):
                with st.expander("View LLM logs"):
                    st.code(msg["logs"], language="text")

    prompt = st.chat_input("Type your finance question...")
    if prompt:
        current_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, logs = _ask_assistant(selected_user_id, prompt)
            st.markdown(answer)
            if logs:
                with st.expander("View LLM logs"):
                    st.code(logs, language="text")

        current_messages.append(
            {"role": "assistant", "content": answer, "logs": logs}
        )


if __name__ == "__main__":
    main()
