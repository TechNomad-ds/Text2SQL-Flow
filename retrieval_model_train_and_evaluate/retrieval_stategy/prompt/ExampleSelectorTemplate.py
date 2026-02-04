import numpy as np
import random
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
import gc
import math
from vllm.inputs.data import TokensPrompt
import torch.nn.functional as F
from tqdm import tqdm
import sqlglot
from sqlglot import exp

from utils.utils import sql2skeleton, jaccard_similarity
from utils.linking_utils.application import mask_question_with_schema_linking


class BasicExampleSelector(object):
    def __init__(self, data, *args, **kwargs):
        self.data = data
        self.train_json = self.data.get_train_json()
        if self.train_json == []:
            print("List is empty.")
            raise ValueError("ERROR: train_json is empty.")
        self.db_ids = [d["db_id"] for d in self.train_json]
        self.train_questions = self.data.get_train_questions()


    def get_examples(self, question, num_example, cross_domain=False):
        pass

    def domain_mask(self, candidates: list, db_id):
        cross_domain_candidates = [candidates[i] for i in range(len(self.db_ids)) if self.db_ids[i] != db_id]
        return cross_domain_candidates

    def retrieve_index(self, indexes: list, db_id):
        cross_domain_indexes = [i for i in range(len(self.db_ids)) if self.db_ids[i] != db_id]
        retrieved_indexes = [cross_domain_indexes[i] for i in indexes]
        return retrieved_indexes


class RandomExampleSelector(BasicExampleSelector):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data)
        random.seed(0)

    def get_examples(self, target, num_example, cross_domain=False):
        train_json = self.train_json
        indexes = list(range(len(train_json)))
        if cross_domain:
            indexes = domain_mask(indexes, target["db_id"])
        selected_indexes = random.sample(indexes, num_example)
        if cross_domain:
            selected_indexes = retrieve_index(selected_indexes, target["db_id"])
        return [train_json[index] for index in selected_indexes]

from sqlglot import parse_one
from zss import simple_distance
from concurrent.futures import ProcessPoolExecutor, as_completed

def compute_similarity(z1, z2, len1, len2):
    # tree edit distance
    d = simple_distance(z1, z2)
    sim = 1 - (d / max(len1, len2))
    return sim

class AstZssSelector(BasicExampleSelector):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data)
        from sqlglot import parse_one
        train_json = self.train_json
        indexes = list(range(len(train_json)))
        self.error = False
        for i in tqdm(range(len(train_json)), desc="Processing train SQL ASTs"):
            try:
                train_ast = parse_one(train_json[i]["query"].replace("`", "'"), read="sqlite")
                # print(f"Processing train SQL AST index {i}")
                train_json[i]["zss_ast"] = self.sqlglot_ast_to_zss(train_ast)
                train_json[i]["ast_len"] = len(list(train_ast.walk()))
            except Exception as e:
                print(f"Error parsing target SQL: {train_json[i]['query']}, error: {e}")
                self.error = True
                
            
    def sqlglot_ast_to_zss(self, node):
        from zss import Node
        label = node.__class__.__name__

        z_node = Node(label)

        for child in node.args.values():
            if hasattr(child, "args"):
                z_node.addkid(self.sqlglot_ast_to_zss(child))
            elif isinstance(child, (list, tuple)):
                for c in child:
                    if hasattr(c, "args"):
                        z_node.addkid(self.sqlglot_ast_to_zss(c))

        return z_node

    def get_examples(self, target, num_example, cross_domain=False):
        if self.error:
            return RandomExampleSelector.get_examples(self, target, num_example, cross_domain=cross_domain)

        target_query = target["query"].replace("`", "'")
        try:
            target_ast = parse_one(target_query)
        except Exception as e:
            print(f"Error parsing target SQL: {target_query}, error: {e}")
            return RandomExampleSelector.get_examples(self, target, num_example, cross_domain=cross_domain)
        z1 = self.sqlglot_ast_to_zss(target_ast)
        target_len = len(list(target_ast.walk()))

        train_json = self.train_json

        train_zss = [train_json[i]["zss_ast"] for i in range(len(train_json))]
        train_lens = [train_json[i]["ast_len"] for i in range(len(train_json))]

        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(compute_similarity, z1, train_zss[i], target_len, train_lens[i])
                for i in range(len(train_json))
            ]

            for i, fut in enumerate(as_completed(futures)):
                try:
                    sim = fut.result()
                except Exception as e:
                    sim = 0.0
                    print(f"Error computing similarity for train index {i}: {e}")
                train_json[i]["ast_similarity"] = sim

                # print(f"AST similarity between target and train index {i}: {sim}")

        pairs = [(train_json[i]["ast_similarity"], i) for i in range(len(train_json))]
        pairs_sorted = sorted(pairs, key=lambda x: x[0], reverse=True)

        top_pairs  = []
        for s, idx in pairs_sorted:
            similar_db_id = train_json[idx]["db_id"]
            if cross_domain and similar_db_id == target["db_id"]:
                continue
            if train_json[idx]["question"] == target["question"]:
                continue
            top_pairs.append((idx, s))
            if len(top_pairs) >= num_example:
                break

        return [train_json[idx] for (idx, s) in top_pairs]


class CosineSimilarExampleSelector(BasicExampleSelector):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data)

        self.SELECT_MODEL = "/path/hf-models/Qwen3-Embedding-0.6B"
        print(f"------select_model:{self.SELECT_MODEL}")
        self.model = LLM(model=self.SELECT_MODEL, task="embed")

        train_embeddings = self.model.embed(self.train_questions)
        self.train_embeddings = torch.tensor([o.outputs.embedding for o in train_embeddings])
        self.train_embeddings = F.normalize(self.train_embeddings, p=2, dim=1)

        
    def get_examples(self, target, num_example, cross_domain=False):
        # target_embedding = self.bert_model.encode([target["question"]])
        target_embedding = self.model.embed([target["question"]])
        target_embedding = torch.tensor([o.outputs.embedding for o in target_embedding])
        target_embedding = F.normalize(target_embedding, p=2, dim=1)
        # target_embedding = self.bert_model.embed_text([target["question"]]).cpu().detach().numpy()

        # find the most similar question in train dataset
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = np.squeeze(cosine_similarity(target_embedding, self.train_embeddings)).tolist()
        pairs = [(similarity, index) for similarity, index in zip(similarities, range(len(similarities)))]
        # print("-----------------cosine similarities-----------------")

        train_json = self.train_json
        pairs_sorted = sorted(pairs, key=lambda x: x[0], reverse=True)
        top_pairs = list()
        for s, index in pairs_sorted:
            similar_db_id = train_json[index]["db_id"]
            if cross_domain and similar_db_id == target["db_id"]:
                continue
            if train_json[index]["question"] == target["question"]:
                continue
            top_pairs.append((index, s))
            if len(top_pairs) >= num_example:
                break

        return [train_json[index] for (index, s) in top_pairs]

class CosineMaskSimilarExampleSelector(BasicExampleSelector):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data)

        self.SELECT_MODEL = "/path/hf-models/Qwen3-Embedding-0.6B"
        print(f"------select_model:{self.SELECT_MODEL}")
        self.model = LLM(model=self.SELECT_MODEL, task="embed")

        self.mask_token = "<mask>"  # the "<mask>" is the mask token of all-mpnet-base-v2
        self.value_token = "<unk>" # the "<unk>" is the unknown token of all-mpnet-base-v2

        train_mask_questions = mask_question_with_schema_linking(self.train_json, mask_tag=self.mask_token, value_tag=self.value_token)
        train_embeddings = self.model.embed(train_mask_questions)
        self.train_embeddings = torch.tensor([o.outputs.embedding for o in train_embeddings])
        self.train_embeddings = F.normalize(self.train_embeddings, p=2, dim=1)

        
    def get_examples(self, target, num_example, cross_domain=False):
        # target_embedding = self.bert_model.encode([target["question"]])
        target_mask_question = mask_question_with_schema_linking([target], mask_tag=self.mask_token, value_tag=self.value_token)
        target_embedding = self.model.embed(target_mask_question)
        target_embedding = torch.tensor([o.outputs.embedding for o in target_embedding])
        target_embedding = F.normalize(target_embedding, p=2, dim=1)
        # target_embedding = self.bert_model.embed_text([target["question"]]).cpu().detach().numpy()

        # find the most similar question in train dataset
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = np.squeeze(cosine_similarity(target_embedding, self.train_embeddings)).tolist()
        pairs = [(similarity, index) for similarity, index in zip(similarities, range(len(similarities)))]
        # print("-----------------cosine similarities-----------------")

        train_json = self.train_json
        pairs_sorted = sorted(pairs, key=lambda x: x[0], reverse=True)
        top_pairs = list()
        for s, index in pairs_sorted:
            similar_db_id = train_json[index]["db_id"]
            if cross_domain and similar_db_id == target["db_id"]:
                continue
            if train_json[index]["question"] == target["question"]:
                continue
            top_pairs.append((index, s))
            if len(top_pairs) >= num_example:
                break

        return [train_json[index] for (index, s) in top_pairs]

class CosineSQLSimilarExampleSelector(BasicExampleSelector):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data)

        self.SELECT_MODEL = "/path/hf-models/Qwen3-Embedding-0.6B"
        print(f"------select_model:{self.SELECT_MODEL}")
        self.model = LLM(model=self.SELECT_MODEL, task="embed")
        train_queries = [self.train_json[d]["query"] for d in range(len(self.train_json))]

        print(f"query examples: {train_queries[:2]}")
        train_embeddings = self.model.embed(train_queries)
        self.train_embeddings = torch.tensor([o.outputs.embedding for o in train_embeddings])
        self.train_embeddings = F.normalize(self.train_embeddings, p=2, dim=1)

        
    def get_examples(self, target, num_example, cross_domain=False):
        # target_embedding = self.bert_model.encode([target["question"]])
        target_embedding = self.model.embed([target["pre_sql"]])
        # print(f"target query: {target['query']}")
        # print(f"masked target query: {target['masked_sql']}")
        target_embedding = torch.tensor([o.outputs.embedding for o in target_embedding])
        target_embedding = F.normalize(target_embedding, p=2, dim=1)
        # target_embedding = self.bert_model.embed_text([target["question"]]).cpu().detach().numpy()

        # find the most similar question in train dataset
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = np.squeeze(cosine_similarity(target_embedding, self.train_embeddings)).tolist()
        pairs = [(similarity, index) for similarity, index in zip(similarities, range(len(similarities)))]
        # print("-----------------cosine similarities-----------------")

        train_json = self.train_json
        pairs_sorted = sorted(pairs, key=lambda x: x[0], reverse=True)
        top_pairs = list()
        for s, index in pairs_sorted:
            similar_db_id = train_json[index]["db_id"]
            if cross_domain and similar_db_id == target["db_id"]:
                continue
            if train_json[index]["question"] == target["question"]:
                continue
            top_pairs.append((index, s))
            if len(top_pairs) >= num_example:
                break

        return [train_json[index] for (index, s) in top_pairs]
    

class CosineMaskSQLSimilarExampleSelector(BasicExampleSelector):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data)

        self.SELECT_MODEL = "/path/hf-models/Qwen3-Embedding-0.6B"
        print(f"------select_model:{self.SELECT_MODEL}")
        self.model = LLM(model=self.SELECT_MODEL, task="embed")
        train_queries = [self.train_json[d]["masked_sql"] for d in range(len(self.train_json))]

        train_embeddings = self.model.embed(train_queries)
        self.train_embeddings = torch.tensor([o.outputs.embedding for o in train_embeddings])
        self.train_embeddings = F.normalize(self.train_embeddings, p=2, dim=1)

        
    def get_examples(self, target, num_example, cross_domain=False):
        # target_embedding = self.bert_model.encode([target["question"]])
        target_embedding = self.model.embed([target["masked_pre_sql"]])
        # print(f"target query: {target['query']}")
        # print(f'masked target query: {target["masked_pre_sql"]}')
        target_embedding = torch.tensor([o.outputs.embedding for o in target_embedding])
        target_embedding = F.normalize(target_embedding, p=2, dim=1)
        # target_embedding = self.bert_model.embed_text([target["question"]]).cpu().detach().numpy()

        # find the most similar question in train dataset
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = np.squeeze(cosine_similarity(target_embedding, self.train_embeddings)).tolist()
        pairs = [(similarity, index) for similarity, index in zip(similarities, range(len(similarities)))]
        # print("-----------------cosine similarities-----------------")

        train_json = self.train_json
        pairs_sorted = sorted(pairs, key=lambda x: x[0], reverse=True)
        top_pairs = list()
        for s, index in pairs_sorted:
            similar_db_id = train_json[index]["db_id"]
            if cross_domain and similar_db_id == target["db_id"]:
                continue
            if train_json[index]["question"] == target["question"]:
                continue
            top_pairs.append((index, s))
            if len(top_pairs) >= num_example:
                break

        return [train_json[index] for (index, s) in top_pairs]


class EuclideanDistanceExampleSelector(BasicExampleSelector):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data)

        self.SELECT_MODEL = "/path/hf-models/Qwen3-Embedding-0.6B"
        print(f"------select_model:{self.SELECT_MODEL}")
        self.model = LLM(model=self.SELECT_MODEL, task="embed")

        train_embeddings = self.model.embed(self.train_questions)
        self.train_embeddings = torch.tensor([o.outputs.embedding for o in train_embeddings])

    def get_examples(self, target, num_example, cross_domain=False):
        # target_embedding = self.bert_model.encode([target["question"]])
        target_embedding = self.model.embed([target["question"]])
        target_embedding = torch.tensor([o.outputs.embedding for o in target_embedding])

        # find the most similar question in train dataset
        from sklearn.metrics.pairwise import euclidean_distances
        distances = np.squeeze(euclidean_distances(target_embedding, self.train_embeddings)).tolist()
        pairs = [(distance, index) for distance, index in zip(distances, range(len(distances)))]

        train_json = self.train_json
        pairs_sorted = sorted(pairs, key=lambda x: x[0])
        top_pairs = list()
        for d, index in pairs_sorted:
            similar_db_id = train_json[index]["db_id"]
            if cross_domain and similar_db_id == target["db_id"]:
                continue
            top_pairs.append((index, d))
            if len(top_pairs) >= num_example:
                break

        return [train_json[index] for (index, d) in top_pairs]


class EuclideanDistanceThresholdExampleSelector(BasicExampleSelector):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data)

        self.SELECT_MODEL = "/path/hf-models/Qwen3-Embedding-0.6B"
        # self.top_distances = list()
        self.threshold = 0.85

        from sentence_transformers import SentenceTransformer
        self.bert_model = SentenceTransformer(self.SELECT_MODEL, device="cpu")
        self.train_embeddings = self.bert_model.encode(self.train_questions)

    def get_examples(self, target, num_example, cross_domain=False):
        target_embedding = self.bert_model.encode([target["question"]])

        # find the most similar question in train dataset
        from sklearn.metrics.pairwise import euclidean_distances
        distances = np.squeeze(euclidean_distances(target_embedding, self.train_embeddings)).tolist()
        pairs = [(distance, index) for distance, index in zip(distances, range(len(distances)))]

        train_json = self.train_json
        pairs_sorted = sorted(pairs, key=lambda x: x[0])
        top_pairs = list()
        for d, index in pairs_sorted:
            similar_db_id = train_json[index]["db_id"]
            if (cross_domain and similar_db_id == target["db_id"]) or d > self.threshold:
                continue
            top_pairs.append((index, d))
            # self.top_distances.append(d)
            if len(top_pairs) >= num_example:
                break
        # print("mean", np.mean(self.top_distances))    # 0.822
        # print("std", np.std(self.top_distances, ddof=1))  # 0.144
        # print("max", max(self.top_distances)) # 1.166

        return [train_json[index] for (index, d) in top_pairs]

class EuclideanDistanceSkeletonSimilarThresholdSelector(BasicExampleSelector):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data)

        self.threshold = 0.85
        
        self.SELECT_MODEL = "/path/hf-models/Qwen3-Embedding-0.6B"
        print(f"------select_model:{self.SELECT_MODEL}")
        self.model = LLM(model=self.SELECT_MODEL, task="embed")
        self.mask_token = "<mask>"  # the "<mask>" is the mask token of all-mpnet-base-v2
        self.value_token = "<unk>" # the "<unk>" is the unknown token of all-mpnet-base-v2

        # train_mask_questions = mask_question_with_schema_linking(self.train_json, mask_tag=self.mask_token, value_tag=self.value_token)
        train_embeddings = self.model.embed(self.train_questions)
        self.train_embeddings = torch.tensor([o.outputs.embedding for o in train_embeddings])
        self.train_embeddings = F.normalize(self.train_embeddings, p=2, dim=1)

    def get_examples(self, target, num_example, cross_domain=False):
        # target_mask_question = mask_question_with_schema_linking([target], mask_tag=self.mask_token, value_tag=self.value_token)
        # target_embedding = self.bert_model.encode(target_mask_question)
        target_embedding = self.model.embed([target["question"]])
        target_embedding = torch.tensor([o.outputs.embedding for o in target_embedding])
        target_embedding = F.normalize(target_embedding, p=2, dim=1)

        # find the most similar question in train dataset
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = np.squeeze(cosine_similarity(target_embedding, self.train_embeddings)).tolist()
        pairs = [(similarity, index) for similarity, index in zip(similarities, range(len(similarities)))]
        # print("-----------------cosine similarities-----------------")

        train_json = self.train_json
        pairs_sorted = sorted(pairs, key=lambda x: x[0], reverse=True)
        top_pairs = list()
        for d, index in pairs_sorted:
            similar_db_id = train_json[index]["db_id"]
            if cross_domain and similar_db_id == target["db_id"]:
                continue
            # Skeleton similarity
            if jaccard_similarity(train_json[index]["query_skeleton"], target["query_skeleton"]) < self.threshold:
                continue
            top_pairs.append((index, d))
            if len(top_pairs) >= num_example:
                break

        if len(top_pairs) < num_example:
            for d, index in pairs_sorted:
                similar_db_id = train_json[index]["db_id"]
                if cross_domain and similar_db_id == target["db_id"]:
                    continue
                # Skeleton similarity
                if jaccard_similarity(train_json[index]["query_skeleton"], target["query_skeleton"]) >= self.threshold:
                    continue
                top_pairs.append((index, d))
                if len(top_pairs) >= num_example:
                    break

        return [train_json[index] for (index, d) in top_pairs]

class EuclideanDistanceSkeletonSimilarThresholdMASKSelector(BasicExampleSelector):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data)

        self.threshold = 0.85
        
        self.SELECT_MODEL = "/path/hf-models/Qwen3-Embedding-0.6B"
        print(f"------select_model:{self.SELECT_MODEL}")
        self.model = LLM(model=self.SELECT_MODEL, task="embed")
        self.mask_token = "<mask>"  # the "<mask>" is the mask token of all-mpnet-base-v2
        self.value_token = "<unk>" # the "<unk>" is the unknown token of all-mpnet-base-v2

        train_mask_questions = mask_question_with_schema_linking(self.train_json, mask_tag=self.mask_token, value_tag=self.value_token)
        train_embeddings = self.model.embed(train_mask_questions)
        self.train_embeddings = torch.tensor([o.outputs.embedding for o in train_embeddings])
        self.train_embeddings = F.normalize(self.train_embeddings, p=2, dim=1)

    def get_examples(self, target, num_example, cross_domain=False):
        target_mask_question = mask_question_with_schema_linking([target], mask_tag=self.mask_token, value_tag=self.value_token)
        # target_embedding = self.bert_model.encode(target_mask_question)
        target_embedding = self.model.embed(target_mask_question)
        target_embedding = torch.tensor([o.outputs.embedding for o in target_embedding])
        target_embedding = F.normalize(target_embedding, p=2, dim=1)

        # find the most similar question in train dataset
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = np.squeeze(cosine_similarity(target_embedding, self.train_embeddings)).tolist()
        pairs = [(similarity, index) for similarity, index in zip(similarities, range(len(similarities)))]
        # print("-----------------cosine similarities-----------------")

        train_json = self.train_json
        pairs_sorted = sorted(pairs, key=lambda x: x[0], reverse=True)
        top_pairs = list()
        for d, index in pairs_sorted:
            similar_db_id = train_json[index]["db_id"]
            if cross_domain and similar_db_id == target["db_id"]:
                continue
            # Skeleton similarity
            if jaccard_similarity(train_json[index]["query_skeleton"], target["query_skeleton"]) < self.threshold:
                continue
            top_pairs.append((index, d))
            if len(top_pairs) >= num_example:
                break

        if len(top_pairs) < num_example:
            for d, index in pairs_sorted:
                similar_db_id = train_json[index]["db_id"]
                if cross_domain and similar_db_id == target["db_id"]:
                    continue
                # Skeleton similarity
                if jaccard_similarity(train_json[index]["query_skeleton"], target["query_skeleton"]) >= self.threshold:
                    continue
                top_pairs.append((index, d))
                if len(top_pairs) >= num_example:
                    break

        return [train_json[index] for (index, d) in top_pairs]


class EuclideanDistanceQuestionMaskEmbedSelector(BasicExampleSelector):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data)

        self.SELECT_MODEL = "/path/hf-models/Qwen3-Embedding-0.6B"
        self.model = LLM(model=self.SELECT_MODEL, task="embed")
        self.mask_token = "<mask>"  # the "<mask>" is the mask token of all-mpnet-base-v2
        self.value_token = "<unk>" # the "<unk>" is the unknown token of all-mpnet-base-v2

        from sentence_transformers import SentenceTransformer
        train_mask_questions = mask_question_with_schema_linking(self.train_json, mask_tag=self.mask_token, value_tag=self.value_token)
        self.bert_model = SentenceTransformer(self.SELECT_MODEL, device="cpu")
        # self.train_embeddings = self.bert_model.encode(train_mask_questions)
        train_embeddings = self.model.embed(train_mask_questions)
        self.train_embeddings = torch.tensor([o.outputs.embedding for o in train_embeddings])
        self.train_embeddings = F.normalize(self.train_embeddings, p=2, dim=1)

    def get_examples(self, target, num_example, cross_domain=False):
        target_mask_question = mask_question_with_schema_linking([target], mask_tag=self.mask_token, value_tag=self.value_token)
        # target_embedding = self.bert_model.encode(target_mask_question)
        target_embedding = self.model.embed(target_mask_question)
        target_embedding = torch.tensor([o.outputs.embedding for o in target_embedding])
        target_embedding = F.normalize(target_embedding, p=2, dim=1)
        # scores = (target_embedding  @ self.train_embeddings.T)
        # scores = scores.squeeze().tolist()
        # pairs = [(score, index) for score, index in zip(scores, range(len(scores)))]

        # find the most similar question in train dataset
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = np.squeeze(cosine_similarity(target_embedding, self.train_embeddings)).tolist()
        pairs = [(similarity, index) for similarity, index in zip(similarities, range(len(similarities)))]
        # print("-----------------cosine similarities-----------------")
        # sort by similarity descending
        pairs_sorted = sorted(pairs, key=lambda x: x[0], reverse=True)
        train_json = self.train_json
        top_pairs = list()
        for d, index in pairs_sorted:
            similar_db_id = train_json[index]["db_id"]
            if cross_domain and similar_db_id == target["db_id"]:
                continue
            top_pairs.append((index, d))
            if len(top_pairs) >= num_example:
                break

        return [train_json[index] for (index, d) in top_pairs]

class EuclideanDistanceQuestionMaskSelector(BasicExampleSelector):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data)

        self.SELECT_MODEL = "/path/hf-models/Qwen3-Embedding-0.6B"
        self.mask_token = "<mask>"  # the "<mask>" is the mask token of all-mpnet-base-v2
        self.value_token = "<unk>" # the "<unk>" is the unknown token of all-mpnet-base-v2

        from sentence_transformers import SentenceTransformer
        train_mask_questions = mask_question_with_schema_linking(self.train_json, mask_tag=self.mask_token, value_tag=self.value_token)
        self.bert_model = SentenceTransformer(self.SELECT_MODEL, device="cpu")
        self.train_embeddings = self.bert_model.encode(train_mask_questions)

    def get_examples(self, target, num_example, cross_domain=False):
        target_mask_question = mask_question_with_schema_linking([target], mask_tag=self.mask_token, value_tag=self.value_token)
        target_embedding = self.bert_model.encode(target_mask_question)

        # find the most similar question in train dataset
        from sklearn.metrics.pairwise import euclidean_distances
        distances = np.squeeze(euclidean_distances(target_embedding, self.train_embeddings)).tolist()
        pairs = [(distance, index) for distance, index in zip(distances, range(len(distances)))]

        train_json = self.train_json
        pairs_sorted = sorted(pairs, key=lambda x: x[0])
        top_pairs = list()
        for d, index in pairs_sorted:
            similar_db_id = train_json[index]["db_id"]
            if cross_domain and similar_db_id == target["db_id"]:
                continue
            top_pairs.append((index, d))
            if len(top_pairs) >= num_example:
                break

        return [train_json[index] for (index, d) in top_pairs]
    
    
class EuclideanDistancePreSkeletonSimilarThresholdSelector(BasicExampleSelector):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data)

        self.threshold = 0.85

        self.SELECT_MODEL = "/path/hf-models/Qwen3-Embedding-0.6B"
        print(f"------select_model:{self.SELECT_MODEL}")
        self.model = LLM(model=self.SELECT_MODEL, task="embed")

        # train_embeddings = self.model.embed(self.train_questions)
        # self.train_embeddings = torch.tensor([o.outputs.embedding for o in train_embeddings])
        # self.train_embeddings = F.normalize(self.train_embeddings, p=2, dim=1)

        train_embeddings = self.model.embed(self.train_questions)
        self.train_embeddings = torch.tensor([o.outputs.embedding for o in train_embeddings])
        self.train_embeddings = F.normalize(self.train_embeddings, p=2, dim=1)

    def get_examples(self, target, num_example, cross_domain=False):
        # target_embedding = self.bert_model.encode([target["question"]])
        target_embedding = self.model.embed([target["question"]])
        target_embedding = torch.tensor([o.outputs.embedding for o in target_embedding])
        target_embedding = F.normalize(target_embedding, p=2, dim=1)

        # find the most similar question in train dataset
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = np.squeeze(cosine_similarity(target_embedding, self.train_embeddings)).tolist()
        pairs = [(similarity, index) for similarity, index in zip(similarities, range(len(similarities)))]
        # print("-----------------cosine similarities-----------------no mask")
        train_json = self.train_json
        pairs_sorted = sorted(pairs, key=lambda x: x[0], reverse=True)
        top_pairs = list()
        for d, index in pairs_sorted:
            similar_db_id = train_json[index]["db_id"]
            if cross_domain and similar_db_id == target["db_id"]:
                continue
            # Skeleton similarity
            # print(f"train_json[index]['pre_skeleton']:{train_json[index]['pre_skeleton']}")
            # print(f"target['pre_skeleton']:{target['pre_skeleton']}")
            if jaccard_similarity(train_json[index]["pre_skeleton"], target["pre_skeleton"]) < self.threshold:
                continue
            top_pairs.append((index, d))
            if len(top_pairs) >= num_example:
                break

        if len(top_pairs) < num_example:
            for d, index in pairs_sorted:
                similar_db_id = train_json[index]["db_id"]
                if cross_domain and similar_db_id == target["db_id"]:
                    continue
                # Skeleton similarity
                if jaccard_similarity(train_json[index]["pre_skeleton"], target["pre_skeleton"]) >= self.threshold:
                    continue
                top_pairs.append((index, d))
                if len(top_pairs) >= num_example:
                    break

        return [train_json[index] for (index, d) in top_pairs]


class EuclideanDistancePreSkeletonSimilarPlusSelector(BasicExampleSelector):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data)

        self.SELECT_MODEL = "/path/hf-models/Qwen3-Embedding-0.6B"

        from sentence_transformers import SentenceTransformer
        self.bert_model = SentenceTransformer(self.SELECT_MODEL, device="cpu")
        self.train_embeddings = self.bert_model.encode(self.train_questions)

    def get_examples(self, target, num_example, cross_domain=False):
        target_embedding = self.bert_model.encode([target["question"]])

        # find the most similar question in train dataset
        from sklearn.metrics.pairwise import euclidean_distances
        distances = np.squeeze(euclidean_distances(target_embedding, self.train_embeddings)).tolist()
        train_json = self.train_json
        for i in range(len(train_json)):
            distances[i] -= jaccard_similarity(train_json[i]["pre_skeleton"], target["pre_skeleton"])
        pairs = [(distance, index) for distance, index in zip(distances, range(len(distances)))]
        pairs_sorted = sorted(pairs, key=lambda x: x[0])
        top_pairs = list()
        for d, index in pairs_sorted:
            similar_db_id = train_json[index]["db_id"]
            if cross_domain and similar_db_id == target["db_id"]:
                continue
            top_pairs.append((index, d))
            if len(top_pairs) >= num_example:
                break

        return [train_json[index] for (index, d) in top_pairs]
    

class EuclideanDistanceQuestionMaskPreSkeletonSimilarThresholdSelector(BasicExampleSelector):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data)

        self.threshold = 0.85

        self.SELECT_MODEL = "/path/hf-models/Qwen3-Embedding-0.6B"
        print(f"------select_model:{self.SELECT_MODEL}")
        self.model = LLM(model=self.SELECT_MODEL, task="embed")
        self.mask_token = "<mask>"  # the "<mask>" is the mask token of all-mpnet-base-v2
        self.value_token = "<unk>" # the "<unk>" is the unknown token of all-mpnet-base-v2

        train_mask_questions = mask_question_with_schema_linking(self.train_json, mask_tag=self.mask_token, value_tag=self.value_token)
        train_embeddings = self.model.embed(train_mask_questions)
        self.train_embeddings = torch.tensor([o.outputs.embedding for o in train_embeddings])
        self.train_embeddings = F.normalize(self.train_embeddings, p=2, dim=1)

    def get_examples(self, target, num_example, cross_domain=False):
        target_mask_question = mask_question_with_schema_linking([target], mask_tag=self.mask_token, value_tag=self.value_token)
        # target_embedding = self.bert_model.encode(target_mask_question)
        target_embedding = self.model.embed(target_mask_question)
        target_embedding = torch.tensor([o.outputs.embedding for o in target_embedding])
        target_embedding = F.normalize(target_embedding, p=2, dim=1)

        # find the most similar question in train dataset
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = np.squeeze(cosine_similarity(target_embedding, self.train_embeddings)).tolist()
        pairs = [(similarity, index) for similarity, index in zip(similarities, range(len(similarities)))]
        # print("-----------------cosine similarities-----------------")
        
        train_json = self.train_json
        pairs_sorted = sorted(pairs, key=lambda x: x[0], reverse=True)
        top_pairs = list()
        for d, index in pairs_sorted:
            similar_db_id = train_json[index]["db_id"]
            if cross_domain and similar_db_id == target["db_id"]:
                continue
            # Skeleton similarity
            if jaccard_similarity(train_json[index]["pre_skeleton"], target["pre_skeleton"]) < self.threshold:
                continue
            top_pairs.append((index, d))
            if len(top_pairs) >= num_example:
                break

        if len(top_pairs) < num_example:
            for d, index in pairs_sorted:
                similar_db_id = train_json[index]["db_id"]
                if cross_domain and similar_db_id == target["db_id"]:
                    continue
                # Skeleton similarity
                if jaccard_similarity(train_json[index]["pre_skeleton"], target["pre_skeleton"]) >= self.threshold:
                    continue
                top_pairs.append((index, d))
                if len(top_pairs) >= num_example:
                    break

        return [train_json[index] for (index, d) in top_pairs]


class EuclideanDistanceQuestionMaskPreSkeletonSimilarThresholdShiftSelector(BasicExampleSelector):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data)

        self.SELECT_MODEL = "/path/hf-models/Qwen3-Embedding-0.6B"
        self.mask_token = "<mask>"  # the "<mask>" is the mask token of all-mpnet-base-v2
        self.value_token = "<unk>"  # the "<unk>" is the unknown token of all-mpnet-base-v2
        self.threshold = 0.85

        from sentence_transformers import SentenceTransformer
        train_mask_questions = mask_question_with_schema_linking(self.train_json, mask_tag=self.mask_token, value_tag=self.value_token)
        self.bert_model = SentenceTransformer(self.SELECT_MODEL, device="cpu")
        self.train_embeddings = self.bert_model.encode(train_mask_questions)

    def get_examples(self, target, num_example, cross_domain=False):
        target_mask_question = mask_question_with_schema_linking([target], mask_tag=self.mask_token, value_tag=self.value_token)
        target_embedding = self.bert_model.encode(target_mask_question)

        # find the most similar question in train dataset
        from sklearn.metrics.pairwise import euclidean_distances
        distances = np.squeeze(euclidean_distances(target_embedding, self.train_embeddings)).tolist()
        pairs = [(distance, index) for distance, index in zip(distances, range(len(distances)))]

        train_json = self.train_json
        pairs_sorted = sorted(pairs, key=lambda x: x[0])
        top_pairs = list()
        for d, index in pairs_sorted:
            similar_db_id = train_json[index]["db_id"]
            if cross_domain and similar_db_id == target["db_id"]:
                continue
            # Skeleton similarity
            if jaccard_similarity(train_json[index]["pre_skeleton"], target["pre_skeleton"]) < self.threshold:
                continue
            top_pairs.append((index, d))
            if len(top_pairs) >= num_example:
                break

        return [train_json[index] for (index, d) in top_pairs]



class EmbedSelector(BasicExampleSelector):
    def __init__(self, data, select_model_path, if_mask, *args, **kwargs):
        super().__init__(data)

        self.SELECT_MODEL = select_model_path
        print(f"self.SELECT_MODEL:{self.SELECT_MODEL}")
        # /path/hf-models/Qwen3-Embedding-0.6B
        # /path/hf-models/Qwen3-Embedding-0.6B-sft
        self.mask_token = "<mask>"  
        self.value_token = "<unk>" 

        from vllm import LLM
        self.model = LLM(model=self.SELECT_MODEL, task="embed")
        train_sqls = []
        self.if_mask = if_mask
        for d in self.train_json:
            # 如果d有query字段
            if 'query' in d:
                sql = d["query"]
            elif 'sql' in d:
                sql = d["sql"]
            else:
                raise KeyError("Data entry is missing a 'query' or 'sql' key:", d)
            if 'masked_sql' in d:
                masked_sql = d['masked_sql']
                # print(f"masked_sql:{masked_sql}")
            else:
                print("Warning: masked_sql not in data entry, using original sql.")
                masked_sql = sql
            if if_mask:
                train_sqls.append(masked_sql)
            else:
                train_sqls.append(sql)
        train_embeddings = self.model.embed(train_sqls)
        self.train_embeddings = torch.tensor([o.outputs.embedding for o in train_embeddings])
        self.train_embeddings = F.normalize(self.train_embeddings, p=2, dim=1)

    def get_examples(self, target, num_example, cross_domain=False):
        if self.if_mask:
            target_mask_question = mask_question_with_schema_linking([target], mask_tag=self.mask_token, value_tag=self.value_token)
            target_embedding = self.model.embed(target_mask_question)
        else:
            target_embedding = self.model.embed([target["question"]])
        target_embedding = torch.tensor([o.outputs.embedding for o in target_embedding])
        target_embedding = F.normalize(target_embedding, p=2, dim=1)
        # train_embeddings = F.normalize(self.train_embeddings, p=2, dim=1)
        train_embeddings = self.train_embeddings

        from sklearn.metrics.pairwise import cosine_similarity
        similarities = np.squeeze(cosine_similarity(target_embedding, self.train_embeddings)).tolist()
        pairs = [(similarity, index) for similarity, index in zip(similarities, range(len(similarities)))]
        # sort by similarity descending
        pairs_sorted = sorted(pairs, key=lambda x: x[0], reverse=True)
        train_json = self.train_json
        top_pairs = list()
        for d, index in pairs_sorted:
            similar_db_id = train_json[index]["db_id"]
            if cross_domain and similar_db_id == target["db_id"]:
                continue
            top_pairs.append((index, d))
            if len(top_pairs) >= num_example:
                break

        return [train_json[index] for (index, d) in top_pairs]

def canonical_sql(e: exp.Expression, dialect: str | None = None) -> str:
    s = e.sql(dialect=dialect, pretty=False, identify=False)
    try:
        return parse_one(s, dialect=dialect).sql(dialect=dialect, pretty=False, identify=False)
    except Exception:
        return s


def compact_ast(
    node: exp.Expression,
    dialect: str | None = None,
    indent: int = 0,
    _cache: dict | None = None,
) -> str:
    if _cache is None:
        _cache = {}

    pad = "  " * indent
    name = node.__class__.__name__

    if isinstance(node, exp.Literal):
        return node.sql(dialect=dialect, pretty=False, identify=False)
    if isinstance(node, exp.Identifier):
        return node.this
    if isinstance(node, exp.DataType):
        return node.sql(dialect=dialect, pretty=False, identify=False)

    def fmt(v, ind):
        if isinstance(v, exp.Expression):
            return compact_ast(v, dialect, ind, _cache)
        if isinstance(v, list):
            if all(isinstance(x, exp.Expression) for x in v):
                inner = ",\n".join(("  " * (ind + 1) + fmt(x, ind + 1)) for x in v)
                return "[\n" + inner + "\n" + ("  " * ind) + "]"
            return "[" + ", ".join(fmt(x, ind) for x in v) + "]"
        if isinstance(v, str):
            return repr(v)
        return repr(v)

    orig_sql = None
    kept: list[tuple[str, object]] = []

    for k, v in node.args.items():
        if v is None:
            continue

        if isinstance(v, exp.Expression) or (isinstance(v, list) and any(isinstance(x, exp.Expression) for x in v)):
            kept.append((k, v))
            continue

        if isinstance(v, bool) and v is False:
            continue

        ck = (node.__class__.__name__, k, type(v).__name__, repr(v))
        drop = _cache.get(ck)

        if drop is None:
            if orig_sql is None:
                orig_sql = canonical_sql(node, dialect)

            test = node.copy()
            try:
                test.set(k, None) 
                drop = (canonical_sql(test, dialect) == orig_sql)
            except Exception:
                drop = False 
            _cache[ck] = drop

        if not drop:
            kept.append((k, v))

    if not kept:
        return f"{name}()"

    lines = [f"{name}("]
    for k, v in kept:
        k2 = "from" if k == "from_" else k
        lines.append(f"{pad}  {k2}={fmt(v, indent + 1)},")
    lines.append(f"{pad})")
    return "\n".join(lines)

SENSITIVE_IDENT_KEYS = {"catalog", "db", "database", "schema", "table", "this"}

def mask_schema(
    root: exp.Expression,
    mask: str = "<mask>",
    *,
    mask_alias: bool = False,
    mask_misquoted_ident: bool = True,
) -> exp.Expression:
    tree = root.copy()

    def _set_ident(node: exp.Expression):
        if isinstance(node, exp.Identifier):
            node.set("this", mask)
            return
        if isinstance(node, exp.Literal) and bool(node.args.get("is_string")):
            node.set("this", mask)
            return

    def _mask_fields(node: exp.Expression, keys: set[str]):
        for k in list(node.args.keys()):
            if k not in keys:
                continue
            v = node.args.get(k)
            if isinstance(v, exp.Expression):
                _set_ident(v)
            elif isinstance(v, str):
                node.set(k, mask)

    LITERAL_IDENT_POS = {
        exp.EQ: {"this"},  
        exp.NEQ: {"this"},
        exp.GT: {"this"},
        exp.GTE: {"this"},
        exp.LT: {"this"},
        exp.LTE: {"this"},
        exp.Like: {"this"},
        exp.ILike: {"this"},
        exp.Cast: {"this"},
        exp.Paren: {"this"},
        exp.Add: {"this", "expression"},
        exp.Sub: {"this", "expression"},
        exp.Mul: {"this", "expression"},
        exp.Div: {"this", "expression"},
        exp.Mod: {"this", "expression"},
    }

    def walk(node: exp.Expression, parent: exp.Expression | None = None, parent_key: str | None = None):
        if isinstance(node, exp.Table):
            _mask_fields(node, {"this", "db", "catalog", "schema"})
        elif isinstance(node, exp.Column):
            _mask_fields(node, {"this", "table", "db", "catalog", "schema"})
        elif isinstance(node, exp.ColumnDef):
            _mask_fields(node, {"this"})  # CREATE TABLE 的列名
        elif isinstance(node, exp.Create):
            _mask_fields(node, {"this", "db", "catalog", "schema"})
        elif isinstance(node, exp.Schema):
            _mask_fields(node, {"this", "db", "catalog", "schema"})

        if mask_alias and isinstance(node, exp.Alias):
            a = node.args.get("alias")
            if isinstance(a, exp.Identifier):
                _set_ident(a)

        if mask_misquoted_ident and isinstance(node, exp.Literal) and bool(node.args.get("is_string")):
            if parent is not None and parent_key is not None:
                ks = LITERAL_IDENT_POS.get(type(parent))
                if ks and parent_key in ks:
                    _set_ident(node)

        for k, v in node.args.items():
            if isinstance(v, exp.Expression):
                walk(v, node, k)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, exp.Expression):
                        walk(item, node, k)

    walk(tree)
    return tree

class ASTEmbedSelector(BasicExampleSelector):
    def __init__(self, data, select_model_path, if_mask, *args, **kwargs):
        super().__init__(data)

        self.SELECT_MODEL = select_model_path
        print(f"self.SELECT_MODEL:{self.SELECT_MODEL}")
        self.mask_token = "<mask>"  
        self.value_token = "<unk>" 

        from vllm import LLM
        self.model = LLM(model=self.SELECT_MODEL, task="embed")
        train_asts = []
        self.if_mask = if_mask
        for d in self.train_json:
            if 'query' in d:
                sql = d["query"]
            elif 'sql' in d:
                sql = d["sql"]
            else:
                raise KeyError("Data entry is missing a 'query' or 'sql' key:", d)
            ast = sqlglot.parse_one(sql, read="sqlite")
            # serialized_ast = compact_ast(ast)
            if if_mask:
                masked = mask_schema(ast, mask="<mask>", mask_alias=False, mask_misquoted_ident=True)
                serialized_ast = compact_ast(masked)
            else:
                serialized_ast = compact_ast(ast)
            serialized = serialized_ast
            train_asts.append(serialized)
        train_embeddings = self.model.embed(train_asts)
        self.train_embeddings = torch.tensor([o.outputs.embedding for o in train_embeddings])
        self.train_embeddings = F.normalize(self.train_embeddings, p=2, dim=1)

    def get_examples(self, target, num_example, cross_domain=False):
        if self.if_mask:
            target_mask_question = mask_question_with_schema_linking([target], mask_tag=self.mask_token, value_tag=self.value_token)
            target_embedding = self.model.embed(target_mask_question)
        else:
            target_embedding = self.model.embed([target["question"]])
        
        target_embedding = torch.tensor([o.outputs.embedding for o in target_embedding])
        target_embedding = F.normalize(target_embedding, p=2, dim=1)
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = np.squeeze(cosine_similarity(target_embedding, self.train_embeddings)).tolist()
        pairs = [(similarity, index) for similarity, index in zip(similarities, range(len(similarities)))]
        # sort by similarity descending
        pairs_sorted = sorted(pairs, key=lambda x: x[0], reverse=True)
        train_json = self.train_json
        top_pairs = list()
        for d, index in pairs_sorted:
            similar_db_id = train_json[index]["db_id"]
            if cross_domain and similar_db_id == target["db_id"]:
                continue
            top_pairs.append((index, d))
            if len(top_pairs) >= num_example:
                break

        return [train_json[index] for (index, d) in top_pairs]
    
class CosineASTSimilarExampleSelector(BasicExampleSelector):
    def __init__(self, data, if_mask, *args, **kwargs):
        super().__init__(data)

        self.SELECT_MODEL = "/path/hf-models/Qwen3-Embedding-0.6B"
        print(f"------select_model:{self.SELECT_MODEL}")
        self.model = LLM(model=self.SELECT_MODEL, task="embed")
        train_asts = []
        self.if_mask = if_mask
        for d in self.train_json:
            # 如果d有query字段
            if 'query' in d:
                sql = d["query"]
            elif 'sql' in d:
                sql = d["sql"]
            else:
                raise KeyError("Data entry is missing a 'query' or 'sql' key:", d)
            ast = sqlglot.parse_one(sql, read="sqlite")
            # serialized_ast = compact_ast(ast)
            if if_mask:
                masked = mask_schema(ast, mask="<mask>", mask_alias=False, mask_misquoted_ident=True)
                serialized_ast = compact_ast(masked)
            else:
                serialized_ast = compact_ast(ast)
            serialized = serialized_ast
            train_asts.append(serialized)
        train_embeddings = self.model.embed(train_asts)
        self.train_embeddings = torch.tensor([o.outputs.embedding for o in train_embeddings])
        self.train_embeddings = F.normalize(self.train_embeddings, p=2, dim=1)

    def get_examples(self, target, num_example, cross_domain=False):
        # target_embedding = self.bert_model.encode([target["question"]])
        target_query = target["pre_sql"].replace("`", "'")
        try:
            ast = sqlglot.parse_one(target_query, read="sqlite")
        except Exception as e:
            print(f"Error parsing target SQL: {target_query}, error: {e}")
            return RandomExampleSelector.get_examples(self, target, num_example, cross_domain=cross_domain)
        # serialized_ast = compact_ast(ast)
        if self.if_mask:
            masked = mask_schema(ast, mask="<mask>", mask_alias=False, mask_misquoted_ident=True)
            serialized_ast = compact_ast(masked)
        else:
            serialized_ast = compact_ast(ast)
        target_embedding = self.model.embed([serialized_ast])
        # print(f"target query: {target['query']}")
        # print(f"masked target query: {target['masked_sql']}")
        target_embedding = torch.tensor([o.outputs.embedding for o in target_embedding])
        target_embedding = F.normalize(target_embedding, p=2, dim=1)
        # target_embedding = self.bert_model.embed_text([target["question"]]).cpu().detach().numpy()

        # find the most similar question in train dataset
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = np.squeeze(cosine_similarity(target_embedding, self.train_embeddings)).tolist()
        pairs = [(similarity, index) for similarity, index in zip(similarities, range(len(similarities)))]
        # print("-----------------cosine similarities-----------------")

        train_json = self.train_json
        pairs_sorted = sorted(pairs, key=lambda x: x[0], reverse=True)
        top_pairs = list()
        for s, index in pairs_sorted:
            similar_db_id = train_json[index]["db_id"]
            if cross_domain and similar_db_id == target["db_id"]:
                continue
            if train_json[index]["question"] == target["question"]:
                continue
            top_pairs.append((index, s))
            if len(top_pairs) >= num_example:
                break

        return [train_json[index] for (index, s) in top_pairs]
    
