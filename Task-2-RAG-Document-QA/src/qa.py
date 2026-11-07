from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


def answer_question(question, vector_store, api_key, model_name):
    # Retrieve the 3 most relevant chunks
    relevant_docs = vector_store.similarity_search(
        question,
        k=3
    )

    # Prepare context with source information
    context_parts = []

    for doc in relevant_docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page")

        if page is not None:
            source_info = f"{source}, Page {page + 1}"
        else:
            source_info = source

        context_parts.append(
            f"Source: {source_info}\n{doc.page_content}"
        )

    context = "\n\n".join(context_parts)

    # Create Gemini chat model
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.2
    )

    # Create the RAG prompt
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a document-based question answering assistant.

Answer questions using ONLY the provided context.
If the answer is not available in the context, say:
"I couldn't find this information in the provided documents."

Do not make up information.
Mention relevant source information when possible.

Context:
{context}"""
        ),
        ("human", "{question}")
    ])

    # Generate answer
    response = (prompt | llm).invoke({
        "context": context,
        "question": question
    })

    answer = response.text
    return answer, relevant_docs