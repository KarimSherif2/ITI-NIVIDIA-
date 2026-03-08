"""
summarize.py
Generates a structured 7-section summary of the uploaded document.
"""

from langchain_core.prompts import ChatPromptTemplate
from rag_pipeline import get_docstore, llm

_prompt = ChatPromptTemplate.from_template("""You are a legal document analyst.

Using ONLY the document excerpts below, produce a concise structured summary
with these sections:
1. **Parties Involved** - who are the contracting parties?
2. **Purpose** - what is the main purpose of this document?
3. **Key Obligations** - what are the main obligations of each party?
4. **Important Dates / Deadlines** - any dates, durations, or deadlines.
5. **Financial Terms** - any payment, fees, or financial conditions.
6. **Termination Conditions** - how can the contract be terminated?
7. **Notable Clauses** - any unusual or important clauses.

If a section is not covered, write "Not specified in the document."

Document Excerpts:
{context}

Summary:""")


def summarize_contract() -> str:
    try:
        docstore = get_docstore()
    except Exception as e:
        return f"⚠️ Could not load the document. Please upload a contract first. ({e})"

    docs = docstore.similarity_search(
        "contract summary purpose parties obligations", k=8
    )

    if not docs:
        return "⚠️ No content found in the document."

    context = "\n\n---\n\n".join([d.page_content for d in docs])
    final_prompt = _prompt.invoke({"context": context})
    return llm.invoke(final_prompt).content
