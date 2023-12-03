import warnings
warnings.filterwarnings('ignore')

import os
import openai
from llama_index import SimpleDirectoryReader
from llama_index.embeddings import HuggingFaceEmbedding

from dotenv import load_dotenv
import numpy as np
import pandas as pd

from llama_index import ServiceContext, VectorStoreIndex, StorageContext
from llama_index.node_parser import SentenceWindowNodeParser
from llama_index.indices.postprocessor import MetadataReplacementPostProcessor
from llama_index.embeddings import HuggingFaceEmbedding

from llama_index.indices.postprocessor import SentenceTransformerRerank
from llama_index import load_index_from_storage

from trulens_eval import Tru
from trulens_eval import OpenAI as fOpenAI
from trulens_eval import Feedback
from trulens_eval import FeedbackMode
from trulens_eval import TruLlama
from trulens_eval.feedback import Groundedness

# loading the openai key
load_dotenv()
open_api_key=os.environ['OPENAI_API_KEY']
openai.api_key=open_api_key

import nest_asyncio

nest_asyncio.apply()

#config embedding
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

#################### Building Senetcen Window Query Engine ####################
def build_sentence_window_index(
    documents,
    llm,
    embed_model= embed_model,
    sentence_window_size=3,
    save_dir="sentence_index",
):
    # create the sentence window node parser w/ default settings
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=sentence_window_size,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    sentence_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        node_parser=node_parser,
    )
    if not os.path.exists(save_dir):
        sentence_index = VectorStoreIndex.from_documents(
            documents, service_context=sentence_context
        )
        sentence_index.storage_context.persist(persist_dir=save_dir)
    else:
        sentence_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
            service_context=sentence_context,
        )

    return sentence_index


def get_sentence_window_query_engine(sentence_index, similarity_top_k=6, rerank_top_n=2):
    # define postprocessors
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="BAAI/bge-reranker-base"
    )

    sentence_window_engine = sentence_index.as_query_engine(
        similarity_top_k=similarity_top_k, node_postprocessors=[postproc, rerank]
    )
    return sentence_window_engine
