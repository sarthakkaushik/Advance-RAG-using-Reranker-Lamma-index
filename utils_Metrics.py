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


########### BUILDING EVALUATIONE METRICS USINg TRUE EVAL ###############




# Reseting the data base
tru = Tru()
tru.reset_database()

#Creating metrics/feedbacks

def create_feedback_func():
    provider = fOpenAI()

    # Metric-1 Answer relevance
    f_qa_relevance = Feedback(
                    provider.relevance_with_cot_reasons,
                    name="Answer Relevance"
                            ).on_input_output()
    
    # Metric 2- Context Relevance
    context_selection = TruLlama.select_source_nodes().node.text

    # Without chain of thought (COT)
    # _qs_relevance = (
    # Feedback(provider.qs_relevance,
    #          name="Context Relevance")
    #                 .on_input()
    #                 .on(context_selection)
    #                 .aggregate(np.mean)
    #                 )

    # With chain of thought (COT)
    f_qs_relevance = (
    Feedback(provider.qs_relevance_with_cot_reasons,
             name="Context Relevance")
    .on_input()
    .on(context_selection)
    .aggregate(np.mean)
                        )
    
    # Metric 3- Groundedness
    grounded = Groundedness(groundedness_provider=provider)
    f_groundedness = (
    Feedback(grounded.groundedness_measure_with_cot_reasons,
             name="Groundedness"
            )
    .on(context_selection)
    .on_output()
    .aggregate(grounded.grounded_statements_aggregator)
    )

    return f_qa_relevance,f_qs_relevance,f_groundedness




def get_prebuilt_trulens_recorder( query_engine,app_id):

    f_qa_relevance,f_qs_relevance,f_groundedness= create_feedback_func()

    tru_recorder = TruLlama(
    query_engine,
    app_id=app_id,
    feedbacks=[
        f_qa_relevance,
        f_qs_relevance,
        f_groundedness
    ]
        )
    
    return tru_recorder

