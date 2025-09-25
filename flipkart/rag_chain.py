"""
RAG Chain Implementation for Product Recommendations
=================================================

This module implements a Retrieval-Augmented Generation (RAG) chain for the product
recommender chatbot. It combines vector similarity search with conversational AI
to provide contextual product recommendations based on user queries.

Key Components:
- History-aware retriever: Maintains conversation context
- Document chain: Combines retrieved reviews with user queries
- Chat history management: Enables multi-turn conversations
- Groq LLM integration: Generates natural language responses

Flow:
1. User query → Context-aware retrieval from vector store
2. Retrieved reviews + query → LLM prompt
3. LLM generates response based on review context
4. Conversation history is maintained for follow-up questions
"""

from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from flipkart.config import Config

class RAGChainBuilder:
    """
    Builder class for creating RAG (Retrieval-Augmented Generation) chains.

    This class constructs a complete RAG pipeline that combines:
    1. Vector similarity search for retrieving relevant product reviews
    2. Conversational AI for generating natural language responses
    3. Chat history management for context-aware conversations

    The RAG chain enables the chatbot to answer product-related queries by:
    - Searching for relevant reviews based on user input
    - Using retrieved reviews as context for LLM response generation
    - Maintaining conversation history for follow-up questions
    """

    def __init__(self, vector_store):
        """
        Initialize the RAG chain builder.

        Args:
            vector_store: AstraDB vector store containing product review embeddings
        """
        self.vector_store = vector_store
        # Initialize Groq LLM with moderate creativity (temperature=0.5)
        self.model = ChatGroq(model=Config.RAG_MODEL, temperature=0.5)
        # Dictionary to store chat histories for different sessions
        self.history_store = {}

    def _get_history(self, session_id: str) -> BaseChatMessageHistory:
        """
        Retrieve or create chat history for a specific session.

        This method manages per-session conversation history, enabling
        context-aware responses in multi-turn conversations.

        Args:
            session_id (str): Unique identifier for the chat session

        Returns:
            BaseChatMessageHistory: Chat history object for the session
        """
        # Create new chat history if session doesn't exist
        if session_id not in self.history_store:
            self.history_store[session_id] = ChatMessageHistory()
        return self.history_store[session_id]
    
    def build_chain(self):
        """
        Build the complete RAG chain with history awareness.

        This method constructs a sophisticated RAG pipeline that:
        1. Creates a history-aware retriever that reformulates queries based on chat context
        2. Sets up a document chain that combines retrieved reviews with user queries
        3. Builds the final RAG chain that orchestrates retrieval and generation
        4. Wraps everything with message history for conversational context

        Returns:
            RunnableWithMessageHistory: Complete RAG chain ready for user interactions
        """
        # Create retriever from vector store - will return top 5 most similar reviews
        # Using more results for better context while maintaining performance
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        # Prompt template for creating context-aware queries
        # This helps reformulate user questions based on previous conversation
        context_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given the chat history and user question, rewrite it as a standalone question."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        # Main prompt template for generating responses based on retrieved reviews
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """You're a helpful e-commerce assistant specializing in product recommendations.

                          Use the provided product reviews and context to answer user questions accurately.

                          Guidelines:
                          - Base your response on the retrieved reviews and product information
                          - Be concise but informative
                          - If asked about products not in the context, mention that you specialize in products from the reviews
                          - Highlight key features, pros, and cons mentioned in reviews
                          - Suggest alternatives when appropriate

                          CONTEXT:\n{context}\n\nQUESTION: {input}"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        # Create history-aware retriever that considers conversation context
        # This reformulates queries based on chat history before retrieving documents
        history_aware_retriever = create_history_aware_retriever(
            self.model, retriever, context_prompt
        )

        # Create document chain that combines retrieved reviews with user query
        # This feeds the context and question to the LLM for response generation
        question_answer_chain = create_stuff_documents_chain(
            self.model, qa_prompt
        )

        # Combine retrieval and generation into complete RAG chain
        # This orchestrates: query → retrieval → context → generation → response
        rag_chain = create_retrieval_chain(
            history_aware_retriever, question_answer_chain
        )

        # Wrap RAG chain with message history for conversational context
        # This enables multi-turn conversations with maintained context
        return RunnableWithMessageHistory(
            rag_chain,                              # The RAG chain to wrap
            self._get_history,                      # Function to get/create chat history
            input_messages_key="input",             # Key for user input in the chain
            history_messages_key="chat_history",    # Key for chat history in prompts
            output_messages_key="answer"            # Key for final response in output
        )


