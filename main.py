import json

import requests
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

from constants import (
    MAX_AGENT_ITERATIONS,
    RAG_RETRIEVER_K,
    AGENT_VERBOSE,
)
from llm import get_llm, get_vectorstore
from prompts import MAIN_SYSTEM_PROMPT, rag_system_prompt, transaction_prompt
from schema import ChatRequest, ChatResponse

load_dotenv()


#  In-Memory Conversation Store  { user_id → [messages] }

conversation_store: dict[int, list] = {}


# ─────────────────────────────────────────────────────────
#  TOOL 1  –  financial_advice_tool
#  Triggers: budgeting, saving tips, financial knowledge
# ─────────────────────────────────────────────────────────

@tool
def financial_advice_tool(query: str) -> str:
    """
    Use this tool when the user asks for FINANCIAL ADVICE, BUDGETING STRATEGIES,
    SAVING TECHNIQUES, or any knowledge-based financial guidance.

    Examples:
    - "What is a good budgeting strategy?"
    - "How can I save more money?"
    - "Give me tips on reducing expenses."
    - "What is the 50/30/20 rule?"
    - "How do I build an emergency fund?"

    Retrieves relevant documents from the FAISS knowledge base and generates
    a grounded response using the RAG pipeline.
    """
    try:
        # Step 1: Retrieve relevant docs from FAISS
        vectorstore = get_vectorstore()
        retriever   = vectorstore.as_retriever(search_kwargs={"k": RAG_RETRIEVER_K})
        docs        = retriever.invoke(query)

        if not docs:
            return (
                "I couldn't find specific information on that topic in my knowledge base. "
                "However, a good general rule is to track your spending, set a budget, "
                "and aim to save at least 20% of your income."
            )

        # Step 2: Build context string from retrieved docs
        context = "\n\n".join(
            [f"[Document {i+1}]\n{doc.page_content}" for i, doc in enumerate(docs)]
        )

        # Step 3: Call RAG LLM with injected context
        rag_llm    = get_llm()
        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", rag_system_prompt(context=context)),
            ("human",  f"{query}"),
        ])
        chain    = rag_prompt | rag_llm
        response = chain.invoke({"query": query})

        return response.content

    except FileNotFoundError as e:
        return (
            f"⚠️ Knowledge base not available: {str(e)}. "
            "Please ensure the FAISS index is set up correctly."
        )
    except Exception as e:
        return f"An error occurred while fetching financial advice: {str(e)}"


# ─────────────────────────────────────────────────────────
#  TOOL 2  –  transactions_tool
#  Triggers: transaction history, spending queries, date filters
# ─────────────────────────────────────────────────────────

@tool
def transactions_tool(user_id: int, query: str) -> str:
    """
    Use this tool when the user asks about their TRANSACTION HISTORY,
    SPENDING DATA, or anything requiring actual financial records.

    Examples:
    - "How much did I spend last week?"
    - "Show me my April transactions."
    - "Were there any transactions on 25 January?"
    - "What did I spend on food this month?"
    - "What is my highest spending category?"
    - "Compare my April and May spending."
    - "Categorize my transactions."

    Fetches all transactions from the Banking API and uses a dedicated
    LLM to analyse and answer the user's specific question.
    """
    try:
        # Step 1: Fetch all transactions from Mock Banking API
        url = "http://127.0.0.1:8000/transactions"

        payload = {
            "user_id": user_id,
            "query": query
        }

        response = requests.get(url, json=payload)

        # Step 2: Serialize transaction data
        financial_data = json.dumps(response.json(), indent=2)

        # Step 3: Build transaction analysis prompt with injected data
        system_prompt = transaction_prompt(financial_data)

        # Step 4: Call Transaction Analysis LLM
        txn_llm    = get_llm()
        txn_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human",  query),
        ])
        chain    = txn_prompt | txn_llm
        response = chain.invoke({"query": query})

        return response.content
    except Exception as e:
        return f"An error occurred while fetching transactions: {str(e)}"


# ─────────────────────────────────────────────────────────
#  Agent Builder  (called once at module import)
# ─────────────────────────────────────────────────────────

def _build_agent() -> AgentExecutor:
    """Builds the LangChain tool-calling agent."""
    llm   = get_llm()
    tools = [financial_advice_tool, transactions_tool]

    prompt = ChatPromptTemplate.from_messages([
        ("system",  MAIN_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human",   "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=AGENT_VERBOSE,
        max_iterations=MAX_AGENT_ITERATIONS,
        handle_parsing_errors=True,
        return_intermediate_steps=False,
    )


# Single agent instance — shared across all requests
_agent_executor = _build_agent()


# ─────────────────────────────────────────────────────────
#  Public Chat Function  (called by api.py)
# ─────────────────────────────────────────────────────────

def chat(request: ChatRequest) -> ChatResponse:
    """
    Processes a user message through the agent and returns a response.
    Maintains per-user conversation history using user_id as the key.

    Args:
        request (ChatRequest): Contains user_id (int) and query (str).

    Returns:
        ChatResponse: Contains user_id, session_id, and the agent's response.
    """
    user_id = request.user_id

    # Initialise conversation history for new users
    if user_id not in conversation_store:
        conversation_store[user_id] = []

    chat_history = conversation_store[user_id]

    # Invoke agent
    result = _agent_executor.invoke({
        "input": f"[user_id: {user_id}] {request.query}",
        "chat_history": chat_history,
    })

    ai_response = result.get("output", "I'm sorry, I couldn't process your request.")

    # Update conversation history
    chat_history.append(HumanMessage(content=request.query))
    chat_history.append(AIMessage(content=ai_response))

    # Keep last 20 messages (10 turns) to avoid token overflow
    if len(chat_history) > 20:
        conversation_store[user_id] = chat_history[-20:]

    return ChatResponse(
        user_id=user_id,
        response=ai_response,
    )


def clear_history(user_id: int) -> bool:
    """Clears conversation history for a given user. Returns True if found."""
    if user_id in conversation_store:
        del conversation_store[user_id]
        return True
    return False