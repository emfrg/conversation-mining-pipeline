"""Step 3: Build Embeddings - Generate vector embeddings for issue reports.

Creates embeddings from issue reports using Vertex AI and stores them in Qdrant
vector database. Embeds concatenated text from canonical question, user problem,
and user goal fields.

Reads:
    config.paths.issue_reports - Structured issue reports from Step 2.

Writes:
    Qdrant collection (config.qdrant.embeddings_collection) - Vector embeddings.
"""

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from config import config
from src.utils.helpers import filter_by_confidence, read_json_file

load_dotenv()


def build_embedding_text(issue: dict) -> str:
    """Build text for embedding from issue report.

    Args:
        issue: Issue report dict with canonical_faq_question, user_problem, user_goal.

    Returns:
        Concatenated text string for embedding.
    """
    return "\n".join(
        [
            f"Canonical FAQ question: {issue['canonical_faq_question']}",
            f"User problem: {issue['user_problem']}",
            f"User goal: {issue['user_goal']}",
        ]
    )


def get_embedding_dimensions() -> int:
    """Get the embedding dimensions from config.

    Returns:
        Embedding dimension size (default 768 if not configured).
    """
    return int(config["embedding"].get("dimensions", 768))


def ensure_collection(
    client: QdrantClient, embedding_dim: int, recreate: bool = False
) -> bool:
    """Ensure Qdrant collection exists with correct dimensions.

    Automatically recreates if dimensions don't match (model changed).

    Args:
        client: Qdrant client instance.
        embedding_dim: Expected embedding dimension size.
        recreate: If True, delete and recreate collection even if it exists.

    Returns:
        True if collection was (re)created, False if existing was kept.
    """
    collection_name = config["qdrant"]["embeddings_collection"]

    if client.collection_exists(collection_name):
        # Check if dimensions match
        collection_info = client.get_collection(collection_name)
        vectors_config = collection_info.config.params.vectors
        assert isinstance(vectors_config, VectorParams), "Expected VectorParams"
        existing_dim = vectors_config.size

        if existing_dim != embedding_dim:
            print(
                f"  Dimension mismatch: collection has {existing_dim}, "
                f"model produces {embedding_dim}. Recreating collection..."
            )
            client.delete_collection(collection_name)
        elif recreate:
            client.delete_collection(collection_name)
        else:
            return False  # collection exists with correct dimensions

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
    )
    return True


def build_documents(issue_reports: dict) -> list[Document]:
    """Convert issue reports to LangChain Documents.

    Args:
        issue_reports: Dict of issue reports keyed by conversation_id.

    Returns:
        List of Document objects with embedding text and full issue metadata.
    """
    documents = []
    for conv_id, issue in issue_reports.items():
        text = build_embedding_text(issue)
        doc = Document(
            page_content=text,  # only this gets embedded
            metadata={"conv_id": conv_id, **issue},
        )
        documents.append(doc)
    return documents


def run() -> int:
    """Run the build_embeddings pipeline step.

    Loads issue reports, generates embeddings, and stores in Qdrant.

    Returns:
        Number of embeddings stored.
    """
    issue_reports_path = config["paths"]["issue_reports"]
    qdrant_host = config["qdrant"]["host"]
    qdrant_port = config["qdrant"]["port"]
    collection_name = config["qdrant"]["embeddings_collection"]
    embedding_model = config["embedding"]["model"]
    recreate = config.get("recreate", False)

    # Skip if collection already exists (unless --recreate)
    if not recreate:
        client = QdrantClient(host=qdrant_host, port=qdrant_port)
        if client.collection_exists(collection_name):
            info = client.get_collection(collection_name)
            print(
                f"  Skipping: Collection '{collection_name}' exists ({info.points_count} points)"
            )
            return int(info.points_count)

    # Load issue reports
    issue_reports = read_json_file(issue_reports_path)
    print(f"  Loaded {len(issue_reports)} issue reports")

    # Filter by confidence if configured
    min_confidence = config.get("filtering", {}).get("min_confidence")
    if min_confidence:
        original_count = len(issue_reports)
        issue_reports = filter_by_confidence(issue_reports, min_confidence)
        print(
            f"  Filtered to {min_confidence}+ confidence: "
            f"{len(issue_reports)}/{original_count} issues"
        )

    # Initialize Vertex AI embeddings with configurable dimensions
    embedding_dim = get_embedding_dimensions()
    embeddings = GoogleGenerativeAIEmbeddings(
        model=embedding_model,
    )
    print(f"  Embedding model: {embedding_model} ({embedding_dim} dimensions)")

    # Initialize Qdrant client and ensure collection exists with correct dimensions
    client = QdrantClient(host=qdrant_host, port=qdrant_port)
    ensure_collection(client, embedding_dim, recreate=True)
    print(f"  Collection ready: {collection_name}")

    # Build documents and add to vector store
    documents = build_documents(issue_reports)
    ids = list(issue_reports.keys())  # Use conversation IDs as document IDs

    # Connect to vector store
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )

    # Add documents to vector store with explicit IDs
    vector_store.add_documents(documents, ids=ids)
    print(f"  Stored {len(documents)} embeddings in Qdrant")

    return len(documents)


if __name__ == "__main__":
    run()
