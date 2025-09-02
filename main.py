import fitz  # PyMuPDF
from langchain_core.documents import Document
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np
from langchain.chat_models import init_chat_model
# from langchain.prompts import PromptTemplate
from langchain.schema.messages import HumanMessage
# from sklearn.metrics.pairwise import cosine_similarity
import base64
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import streamlit as st


@st.cache_resource
def initialize_clip_model():
    """Initializes and returns the CLIP model and processor."""
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    return model, processor


@st.cache_resource
def initialize_llm():
    """Initializes and returns the gemma3:4b"""
    return init_chat_model("ollama:gemma3:4b")


### Embedding functions
def embed_image(image_data, processor, model):
    """Embed image using CLIP"""
    if isinstance(image_data, str):  # If path
        image = Image.open(image_data).convert("RGB")
    else:  # If PIL Image
        image = image_data

    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = model.get_image_features(**inputs)
        # Normalize embeddings to unit vector
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()


def embed_text(text, processor, model):
    """Embed text using CLIP."""
    inputs = processor(
        text=text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77  # CLIP's max token length
    )
    with torch.no_grad():
        features = model.get_text_features(**inputs)
        # Normalize embeddings
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()


def process_pdf_document(pdf_path, clip_processor, clip_model):
    """Processes a PDF to extract and embed text and images."""
    doc = fitz.open(pdf_path)
    all_docs = []
    all_embeddings = []
    image_data_store = {}  # Store actual image data for LLM

    # Text splitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    for i, page in enumerate(doc):
        # process text
        text = page.get_text()
        if text.strip():
            # create temporary document for splitting
            temp_doc = Document(page_content=text, metadata={"page": i, "type": "text"})
            text_chunks = splitter.split_documents([temp_doc])

            # Embed each chunk using CLIP
            for chunk in text_chunks:
                embedding = embed_text(chunk.page_content, clip_processor, clip_model)
                all_embeddings.append(embedding)
                all_docs.append(chunk)

        # process images
        # Three Important Actions:
        # 1. Convert PDF image to PIL format
        # 2. Store as base64 for the model (which needs base64 images)
        # 3. Create CLIP embedding for retrieval
        for img_index, img in enumerate(page.get_images(full=True)):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]

                # Convert to PIL Image
                pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                # Create unique identifier
                image_id = f"page_{i}_img_{img_index}"

                # Store image as base64 for later use with the LLM
                buffered = io.BytesIO()
                pil_image.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                image_data_store[image_id] = img_base64

                # Embed image using CLIP
                embedding = embed_image(pil_image, clip_processor, clip_model)
                all_embeddings.append(embedding)

                # Create document for image
                image_doc = Document(
                    page_content=f"[Image: {image_id}]",
                    metadata={"page": i, "type": "image", "image_id": image_id}
                )
                all_docs.append(image_doc)

            except Exception as e:
                print(f"Error al procesar la imagen {img_index} en la página {i}: {e}")
                continue

    doc.close()
    return all_docs, all_embeddings, image_data_store


def create_vector_store(docs, embeddings):
    """Creates a FAISS vector store from documents and their embeddings."""
    embeddings_array = np.array(embeddings)
    vector_store = FAISS.from_embeddings(
        text_embeddings=[(doc.page_content, emb) for doc, emb in zip(docs, embeddings_array)],
        embedding=None,  # We're using precomputed embeddings
        metadatas=[doc.metadata for doc in docs]
    )
    return vector_store


def retrieve_multimodal(query, vector_store, clip_processor, clip_model, k=2):
    """Unified retrieval using CLIP embeddings for both text and images."""
    # Embed query using CLIP
    query_embedding = embed_text(query, clip_processor, clip_model)

    # Search in unified vector store
    results = vector_store.similarity_search_by_vector(
        embedding=query_embedding,
        k=k
    )

    return results


def create_multimodal_message(query, retrieved_docs, image_data_store):
    """Create a message with both text and images for the model with the querry."""
    content = [{
        "type": "text",
        "text": f"Eres un asistente de restaurante servicial. Un cliente pregunta: '{query}'. Usa la siguiente información del menú para responder."
    }]

    # Separate text and image documents
    text_docs = [doc for doc in retrieved_docs if doc.metadata.get("type") == "text"]
    image_docs = [doc for doc in retrieved_docs if doc.metadata.get("type") == "image"]

    # Add text context
    if text_docs:
        text_context = "\n\n".join([
            f"Elemento del menú (de la página {doc.metadata['page']}): {doc.page_content}"
            for doc in text_docs
        ])
        content.append({
            "type": "text",
            "text": f"\n--- Descripciones del Menú ---\n{text_context}\n"
        })

    # Add images
    for doc in image_docs:
        image_id = doc.metadata.get("image_id")
        if image_id and image_id in image_data_store:
            content.append({
                "type": "text",
                "text": f"\n[Imagen de la página {doc.metadata['page']}]:\n"
            })
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_data_store[image_id]}"
                }
            })

    # Add instruction
    content.append({
        "type": "text",
        "text": "\n\nPor favor, responde la pregunta del cliente basándote en las descripciones, precios e imágenes del menú proporcionadas. Si preguntan cómo se ve un plato o qué contiene, describe la imagen."
    })

    return HumanMessage(content=content)


def multimodal_pdf_rag_pipeline(query, vector_store, image_data_store, llm, clip_processor, clip_model):
    """Main pipeline for multimodal RAG."""
    # Retrieve relevant documents
    context_docs = retrieve_multimodal(query, vector_store, clip_processor, clip_model, k=2)

    # Create multimodal message
    message = create_multimodal_message(query, context_docs, image_data_store)

    # Get response from model
    response = llm.invoke([message])

    # Find the most relevant image from context to display
    image_to_display = None
    for doc in context_docs:
        if doc.metadata.get("type") == "image":
            image_id = doc.metadata.get("image_id")
            if image_id in image_data_store:
                image_to_display = image_data_store[image_id]
                break  # Display the first (most relevant) image found

    # Print retrieved context info
    with st.expander("Contexto recuperado"):
        st.write(f"Se recuperaron {len(context_docs)} documentos:")
        for doc in context_docs:
            doc_type = doc.metadata.get("type", "unknown")
            page = doc.metadata.get("page", "?")
            if doc_type == "text":
                preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                st.write(f"  - Texto de la página {page}: {preview}")
            else:
                st.write(f"  - Imagen de la página {page}")

    return response.content, image_to_display


@st.cache_resource
def load_data(pdf_path, _clip_processor, _clip_model):
    """Loads and processes the PDF, creating a vector store."""
    all_docs, all_embeddings, image_data_store = process_pdf_document(pdf_path, _clip_processor, _clip_model)
    vector_store = create_vector_store(all_docs, all_embeddings)
    return vector_store, image_data_store


def main():
    st.title("Asistente de Restaurante 🍽️")

    # Initialize models
    clip_model, clip_processor = initialize_clip_model()
    llm = initialize_llm()

    # Process the PDF to get documents, embeddings, and image data
    pdf_path = "menu.pdf"
    vector_store, image_data_store = load_data(pdf_path, clip_processor, clip_model)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "¡Bienvenido! Pregúntame cualquier cosa sobre nuestro menú."}]

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if isinstance(message["content"], dict):
                st.markdown(message["content"]["text"])
                if "image" in message["content"] and message["content"]["image"]:
                    st.image(io.BytesIO(base64.b64decode(message["content"]["image"])))
            else:
                st.markdown(message["content"])

    # React to user input
    if query := st.chat_input("¿Qué te gustaría saber?"):
        # Display user message in chat message container
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            with st.spinner("Buscando la mejor respuesta..."):
                answer, image_base64 = multimodal_pdf_rag_pipeline(query, vector_store, image_data_store, llm, clip_processor, clip_model)
                st.markdown(answer)
                if image_base64:
                    st.image(io.BytesIO(base64.b64decode(image_base64)))

        # Add assistant response to chat history
        response_content = {"text": answer}
        if image_base64:
            response_content["image"] = image_base64
        st.session_state.messages.append({"role": "assistant", "content": response_content})


if __name__ == "__main__":
    main()
