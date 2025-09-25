# Flask web application for product recommendation chatbot
# Uses RAG (Retrieval-Augmented Generation) to answer product queries based on Flipkart reviews

from flask import render_template, Flask, request, Response
from prometheus_client import Counter, generate_latest
from flipkart.data_ingestion import DataIngestor
from flipkart.rag_chain import RAGChainBuilder

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Prometheus metric to track total HTTP requests
REQUEST_COUNT = Counter("http_requests_total", "Total HTTP Request")

def create_app():
    """
    Create and configure the Flask application.

    Returns:
        Flask: Configured Flask application instance
    """
    app = Flask(__name__)

    # Initialize vector store with existing data (load_existing=True for faster startup)
    # This connects to AstraDB and loads the pre-ingested product review embeddings
    vector_store = DataIngestor().ingest(load_existing=True)

    # Build RAG chain for conversational product recommendations
    # Combines retrieval from vector store with LLM generation
    rag_chain = RAGChainBuilder(vector_store).build_chain()

    @app.route("/")
    def index():
        """
        Serve the main chat interface page.

        Returns:
            str: Rendered HTML template for the chatbot interface
        """
        # Increment request counter for monitoring
        REQUEST_COUNT.inc()
        return render_template("index.html")

    @app.route("/get", methods=["POST"])
    def get_response():
        """
        Handle chat messages and return AI-generated responses.

        Expects:
            msg (str): User's message from the chat interface

        Returns:
            str: AI-generated response based on product review context
        """
        # Extract user message from form data
        user_input = request.form["msg"]

        # Invoke RAG chain to generate response
        # Uses session-based chat history for context-aware conversations
        response = rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": "user-session"}}
        )["answer"]

        return response

    @app.route("/metrics")
    def metrics():
        """
        Expose Prometheus metrics endpoint for monitoring.

        Returns:
            Response: Prometheus metrics in text format
        """
        return Response(generate_latest(), mimetype="text/plain")

    return app

# Run the application in development mode
if __name__ == "__main__":
    # Create Flask app instance
    app = create_app()
    # Start development server on all interfaces, port 3000 with debug mode enabled
    app.run(host="0.0.0.0", port=3000, debug=True)