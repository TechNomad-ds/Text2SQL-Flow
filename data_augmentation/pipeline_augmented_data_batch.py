from dataflow import get_logger
import argparse

from dataflow.operators.text2sql import (
    SQLVariationGenerator,
    Text2SQLQuestionGenerator,
    Text2SQLPromptGenerator,
    Text2SQLCoTGenerator
)
from dataflow.operators.text2sql import (
    SQLConsistencyFilter,
    SQLExecutabilityFilter
)
from dataflow.prompts.text2sql import (
    SQLConsistencyFilterPrompt,
    Text2SQLCotGeneratorPrompt,
    Text2SQLQuestionGeneratorPrompt,
    SQLVariationGeneratorPrompt,
    Text2SQLPromptGeneratorPrompt
)

from dataflow.pipeline import StreamBatchedPipelineABC
from dataflow.utils.storage import StreamBatchedFileStorage
from dataflow.serving import APILLMServing_request
from dataflow.utils.text2sql.database_manager import DatabaseManager
from dataflow.utils.metrics_collector import MetricsCollector, set_metrics_collector
import os
from typing import Optional, Union


class SQLAugmentationGeneration_APIPipeline(StreamBatchedPipelineABC):
    def __init__(self, db_root_path: str, entry_file_name: str, num_variations: int = 3, model_name: str = "gpt-4o", cache_path: str = None, data_source: str = "original"):
        super().__init__()
        self.logger = get_logger()
        self.db_root_path = db_root_path
        self.model_name = model_name

        cache_root = cache_path or f"./cache_{entry_file_name.split('/')[-1].split('.')[0]}"
        self.storage = StreamBatchedFileStorage(
            first_entry_file_name=entry_file_name,
            cache_path=cache_root,
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl"
        )

        self.llm_serving = APILLMServing_request(
            api_url="https://api.openai.com/v1/chat/completions",
            model_name=model_name,
            max_workers=100 # Concurrency level, adjust according to the situation
        )

        embedding_serving = APILLMServing_request(
            api_url="https://api.openai.com/v1/embeddings",
            model_name="text-embedding-ada-002",
            max_workers=100 # Concurrency level, adjust according to the situation
        )

        database_manager = DatabaseManager(
            db_type="sqlite",
            config={
                "root_path": self.db_root_path
            }
        )

        self.sql_executability_filter_step1 = SQLExecutabilityFilter(
            database_manager=database_manager
        )

        self.sql_variation_generator_step2 = SQLVariationGenerator(
            llm_serving=self.llm_serving,
            database_manager=database_manager,
            num_variations=num_variations, # Number of variations to generate for each SQL
            prompt_template=SQLVariationGeneratorPrompt()
        )

        self.sql_executability_filter_step3 = SQLExecutabilityFilter(
            database_manager=database_manager
        )

        self.text2sql_question_generator_step4 = Text2SQLQuestionGenerator(
            llm_serving=self.llm_serving,
            embedding_serving=embedding_serving,
            database_manager=database_manager,
            question_candidates_num=3,
            prompt_template=Text2SQLQuestionGeneratorPrompt()
        )

        self.sql_consistency_filter_step5 = SQLConsistencyFilter(
            llm_serving=self.llm_serving,
            database_manager=database_manager,
            prompt_template=SQLConsistencyFilterPrompt()
        )     

        self.text2sql_prompt_generator_step6 = Text2SQLPromptGenerator(
            database_manager=database_manager,
            prompt_template=Text2SQLPromptGeneratorPrompt()
        )

        self.sql_cot_generator_step7 = Text2SQLCoTGenerator(
            llm_serving=self.llm_serving,
            database_manager=database_manager,
            prompt_template=Text2SQLCotGeneratorPrompt()
        )

        self.sql_cot_voting_generator_step8 = Text2SQLCoTVotingGenerator(
            database_manager=database_manager
        )

    def forward(self):
        sql_key = "SQL"
        db_id_key = "db_id"
        question_key = "question"
        evidence_key = "evidence"

        self.sql_executability_filter_step1.run(
            storage=self.storage.step(),
            input_sql_key=sql_key,
            input_db_id_key=db_id_key
        )

        self.sql_variation_generator_step2.run(
            storage=self.storage.step(),
            input_sql_key=sql_key,
            input_db_id_key=db_id_key,
            output_sql_variation_type_key="sql_variation_type"
        )

        self.sql_executability_filter_step3.run(
            storage=self.storage.step(),
            input_sql_key=sql_key,
            input_db_id_key=db_id_key
        )

        self.text2sql_question_generator_step4.run(
            storage=self.storage.step(),
            input_sql_key=sql_key,
            input_db_id_key=db_id_key,
            output_question_key=question_key,
            output_evidence_key=evidence_key
        )

        self.sql_consistency_filter_step5.run(
            storage=self.storage.step(),   
            input_sql_key=sql_key,
            input_db_id_key=db_id_key,
            input_question_key=question_key,
            input_evidence_key=evidence_key
        )

        self.text2sql_prompt_generator_step6.run(
            storage=self.storage.step(),
            input_question_key=question_key,
            input_db_id_key=db_id_key,
            input_evidence_key=evidence_key,
            output_prompt_key="prompt"
        )

        self.sql_cot_generator_step7.run(
            storage=self.storage.step(),
            input_sql_key=sql_key,
            input_question_key=question_key,
            input_db_id_key=db_id_key,
            input_evidence_key=evidence_key,
            output_cot_key="cot_reasoning"
        )

        self.sql_cot_voting_generator_step8.run(
            storage=self.storage.step(),
            input_cot_responses_key="cot_responses",
            input_db_id_key=db_id_key,
            output_cot_key="cot_reasoning"
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_root_path", type=str, default="")
    parser.add_argument("--entry_file_name", type=str, default="")
    parser.add_argument("--num_variations", type=int, default=3)
    parser.add_argument("--model_name", type=str, default="gpt-4o")
    parser.add_argument("--cache_path", type=str, default=None)
    parser.add_argument("--data_source", type=str, default="original")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--resume_from_last", action="store_true")
    args = parser.parse_args()

    db_root_path = args.db_root_path
    if isinstance(db_root_path, str) and "," in db_root_path:
        db_root_path = [p for p in (s.strip() for s in db_root_path.split(",")) if p]

    model = SQLAugmentationGeneration_APIPipeline(
        db_root_path=db_root_path,
        entry_file_name=args.entry_file_name,
        num_variations=args.num_variations,
        model_name=args.model_name,
        cache_path=args.cache_path,
        data_source=args.data_source
    )
    model.compile()
    model.forward(batch_size=args.batch_size, resume_from_last=args.resume_from_last)
