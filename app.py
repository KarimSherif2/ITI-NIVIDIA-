"""
app.py
Smart Contract Q&A Assistant - Gradio UI
"""

import gradio as gr
from ingest import ingest_document
from rag_pipeline import load_rag, reload_vectorstore
from utils import format_sources
from summarize import summarize_contract
from guardrails import check_input, check_output

qa_chain = None
chat_history = []  # list of {"role": "user"/"assistant", "content": str}

DISCLAIMER = (
    "⚠️ **Legal Disclaimer:** This tool provides informational assistance only "
    "and does **not** constitute legal advice. Always consult a qualified legal "
    "professional. This system supports **English-language documents only**."
)


def history_for_rag():
    """Convert chat_history dicts to (user, bot) tuples for rag_pipeline memory."""
    pairs = []
    i = 0
    while i < len(chat_history):
        if chat_history[i]["role"] == "user":
            u = chat_history[i]["content"]
            b = (chat_history[i + 1]["content"]
                 if i + 1 < len(chat_history) and chat_history[i + 1]["role"] == "assistant"
                 else "")
            pairs.append((u, b))
            i += 2
        else:
            i += 1
    return pairs


def upload_contract(file):
    global qa_chain, chat_history
    if file is None:
        return "⚠️ Please select a file before clicking Process."
    try:
        ingest_document(file.name)
        reload_vectorstore()
        qa_chain = load_rag()
        chat_history = []
        return "✅ Document processed successfully! Switch to the Chat tab."
    except Exception as e:
        return f"❌ Error processing document: {e}"


def ask_question(question):
    global chat_history

    if qa_chain is None:
        chat_history = chat_history + [
            {"role": "assistant", "content": "⚠️ Please upload and process a contract first."}
        ]
        return chat_history, ""

    if not question or question.strip() == "":
        return chat_history, ""

    # Input safety guardrail
    is_safe, block_reason = check_input(question)
    if not is_safe:
        chat_history = chat_history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": f"🚫 {block_reason}"}
        ]
        return chat_history, ""

    try:
        result = qa_chain({"query": question, "history": history_for_rag()})
        answer = result["result"]
        source_docs = result["source_documents"]

        # Output grounding guardrail
        answer, _ = check_output(answer, source_docs)

        sources = format_sources(source_docs)
        final_answer = f"{answer}\n\n📄 Sources: {' | '.join(sources)}"

        chat_history = chat_history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": final_answer}
        ]
    except Exception as e:
        chat_history = chat_history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": f"❌ Error generating answer: {e}"}
        ]

    return chat_history, ""


def summarize():
    global chat_history
    summary = summarize_contract()
    chat_history = chat_history + [
        {"role": "assistant", "content": f"📋 **Contract Summary**\n\n{summary}"}
    ]
    return chat_history


def show_history():
    if not chat_history:
        return "No history yet."
    lines = []
    for msg in chat_history:
        lines.append(f"{msg['role'].upper()}:\n{msg['content']}")
        lines.append("─" * 60)
    return "\n\n".join(lines)


def save_history():
    if not chat_history:
        return "⚠️ No chat history to save."
    try:
        with open("chat_history.txt", "w", encoding="utf-8") as f:
            for msg in chat_history:
                f.write(f"{msg['role'].upper()}:\n{msg['content']}\n\n{'─' * 60}\n\n")
        return "✅ Chat history saved to chat_history.txt"
    except Exception as e:
        return f"❌ Failed to save: {e}"


def clear_chat():
    global chat_history
    chat_history = []
    return [], ""


# ── UI ─────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Smart Contract Assistant") as app:

    gr.Markdown("# 📄 Smart Contract Q&A Assistant")
    gr.Markdown(DISCLAIMER)

    with gr.Tab("📁 Upload Contract"):
        gr.Markdown("### Step 1: Upload your contract")
        gr.Markdown("**Supported formats:** PDF, DOCX &nbsp;|&nbsp; **Language:** English only")
        file = gr.File(label="Select PDF or DOCX", file_types=[".pdf", ".docx"])
        upload_button = gr.Button("⚙️ Process Document", variant="primary")
        upload_status = gr.Textbox(label="Status", interactive=False)
        upload_button.click(upload_contract, inputs=file, outputs=upload_status)

    with gr.Tab("💬 Chat"):
        gr.Markdown("### Step 2: Ask questions about your contract")
        chatbot = gr.Chatbot(height=500, label="Contract Assistant")
        with gr.Row():
            question = gr.Textbox(
                label="Your question",
                placeholder="e.g. What does Box 1 represent?",
                scale=4
            )
            ask_button = gr.Button("Ask ➤", variant="primary", scale=1)

        ask_button.click(ask_question, inputs=question, outputs=[chatbot, question])
        question.submit(ask_question, inputs=question, outputs=[chatbot, question])

        with gr.Row():
            summary_button = gr.Button("📋 Summarize Contract")
            clear_button = gr.Button("🗑️ Clear Chat")

        summary_button.click(summarize, inputs=None, outputs=chatbot)
        clear_button.click(clear_chat, inputs=None, outputs=[chatbot, question])

        with gr.Accordion("🗂️ Chat History Tools", open=False):
            history_button = gr.Button("Show Chat History")
            history_output = gr.Textbox(label="Chat History", lines=10, interactive=False)
            history_button.click(show_history, inputs=None, outputs=history_output)
            save_button = gr.Button("💾 Save Chat History to File")
            save_status = gr.Textbox(label="Save Status", interactive=False)
            save_button.click(save_history, inputs=None, outputs=save_status)

app.launch(share=False)
