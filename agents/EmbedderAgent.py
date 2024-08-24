# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from abc import ABC
import json
from .core import Agent 
from vertexai.language_models import TextEmbeddingModel,TextEmbeddingInput
from utilities import VECTOR_EMBEDDING_MODEL

import logging
logger = logging.getLogger(__name__)


class EmbedderAgent(Agent, ABC):
    """
    An agent specialized in generating text embeddings using Large Language Models (LLMs).

    This agent supports two modes for generating embeddings:

    1. "vertex": Directly interacts with the Vertex AI TextEmbeddingModel.
    2. "lang-vertex": Uses LangChain's VertexAIEmbeddings for a streamlined interface.

    Attributes:
        agentType (str): Indicates the type of agent, fixed as "EmbedderAgent".
        mode (str): The embedding generation mode ("vertex" or "lang-vertex").
        model: The underlying embedding model (Vertex AI TextEmbeddingModel or LangChain's VertexAIEmbeddings).

    Methods:
        create(question) -> list:
            Generates text embeddings for the given question(s).

            Args:
                question (str or list): The text input for which embeddings are to be generated. Can be a single string or a list of strings.

            Returns:
                list: A list of embedding vectors. Each embedding vector is represented as a list of floating-point numbers.

            Raises:
                ValueError: If the input `question` is not a string or list, or if the specified `mode` is invalid.
    """


    agentType: str = "EmbedderAgent"

    def __init__(self, mode, embeddings_model=VECTOR_EMBEDDING_MODEL): 
        if mode == 'vertex': 
            self.mode = mode 
            self.model = TextEmbeddingModel.from_pretrained(embeddings_model)
            self.task = "RETRIEVAL_DOCUMENT"

        elif mode == 'lang-vertex': 
            self.mode = mode 
            from langchain.embeddings import VertexAIEmbeddings
            self.model = VertexAIEmbeddings() 

        else: raise ValueError('EmbedderAgent mode must be either vertex or lang-vertex')
        
        logger.info(f"Using embedding model {embeddings_model}")



    def create(self, text): 
        """Text embedding with a Large Language Model."""

        if self.mode == 'vertex': 
            if isinstance(text, str): 
                inputs = [TextEmbeddingInput(text, self.task)]
                embeddings = self.model.get_embeddings(inputs)
                for embedding in embeddings:
                    vector = embedding.values
                return vector
            
            elif isinstance(text, list):  
                vector = list() 
                for t in text: 
                    inputs = [TextEmbeddingInput(t, self.task)]
                    embeddings = self.model.get_embeddings(inputs)

                    for embedding in embeddings:
                        vector.append(embedding.values) 
                return vector
            
            elif isinstance(text, dict):
                # Assuming the JSON is a single string, extract the content
                text_content = text.get('content', '')
                inputs = [TextEmbeddingInput(text_content, self.task)]
                embeddings = self.model.get_embeddings(inputs)
                for embedding in embeddings:
                    vector = embedding.values
                return vector

            else: raise ValueError('Input must be either str, list or dict')
            

        elif self.mode == 'lang-vertex': 
            vector = self.embeddings_service.embed_documents(text)
            return vector           