from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate


def create_qa_chain(vectorstore):
    llm = OllamaLLM(model="mistral", temperature=0)

    retriever = vectorstore.as_retriever()

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a helpful AI assistant.
Answer the question ONLY using the provided context.
If the answer is not present, say "I don't know."

Context:
{context}

Question:
{question}

Answer:
""",
    )

    def qa_fn(question: str):
        docs = retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in docs)

        final_prompt = prompt.format(context=context, question=question)

        return llm.invoke(final_prompt)

    return qa_fn
