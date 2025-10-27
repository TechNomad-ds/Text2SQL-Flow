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

from utils.utils import sql2skeleton, jaccard_similarity
from utils.linking_utils.application import mask_question_with_schema_linking


class BasicExampleSelector(object):
    def __init__(self, data, *args, **kwargs):
        self.data = data
        self.train_json = self.data.get_train_json()
        if self.train_json == []:
            print("The list is empty.")
            raise ValueError("Error: training data 'train_json' must not be empty.")
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


class CosineSimilarExampleSelector(BasicExampleSelector):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data)
        self.SELECT_MODEL = "/models/hf-models/Qwen3-Embedding-0.6B"
        print(f"------select_model:{self.SELECT_MODEL}")
        self.model = LLM(model=self.SELECT_MODEL, task="embed")

        train_embeddings = self.model.embed(self.train_questions)
        self.train_embeddings = torch.tensor([o.outputs.embedding for o in train_embeddings])

    def get_examples(self, target, num_example, cross_domain=False):
        target_embedding = self.model.embed([target["question"]])
        target_embedding = torch.tensor([o.outputs.embedding for o in target_embedding])

        # find the most similar question in train dataset
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = np.squeeze(cosine_similarity(target_embedding, self.train_embeddings)).tolist()
        pairs = [(similarity, index) for similarity, index in zip(similarities, range(len(similarities)))]
        print("-----------------cosine similarities-----------------")

        train_json = self.train_json
        pairs_sorted = sorted(pairs, key=lambda x: x[0], reverse=True)
        top_pairs = []
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

        self.SELECT_MODEL = "/models/hf-models/all-mnpet-base-v2"
        print(f"------select_model:{self.SELECT_MODEL}")
        self.model = LLM(model=self.SELECT_MODEL, task="embed")

        train_embeddings = self.model.embed(self.train_questions)
        self.train_embeddings = torch.tensor([o.outputs.embedding for o in train_embeddings])

    def get_examples(self, target, num_example, cross_domain=False):
        target_embedding = self.model.embed([target["question"]])
        target_embedding = torch.tensor([o.outputs.embedding for o in target_embedding])

        # find the most similar question in train dataset
        from sklearn.metrics.pairwise import euclidean_distances
        distances = np.squeeze(euclidean_distances(target_embedding, self.train_embeddings)).tolist()
        pairs = [(distance, index) for distance, index in zip(distances, range(len(distances)))]

        train_json = self.train_json
        pairs_sorted = sorted(pairs, key=lambda x: x[0])
        top_pairs = []
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

        self.SELECT_MODEL = "/models/hf-models/all-mpnet-base-v2"
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
        top_pairs = []
        for d, index in pairs_sorted:
            similar_db_id = train_json[index]["db_id"]
            if (cross_domain and similar_db_id == target["db_id"]) or d > self.threshold:
                continue
            top_pairs.append((index, d))
            if len(top_pairs) >= num_example:
                break

        return [train_json[index] for (index, d) in top_pairs]


class EuclideanDistanceSkeletonSimilarThresholdSelector(BasicExampleSelector):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data)

        self.threshold = 0.85
        self.SELECT_MODEL = "/models/hf-models/Qwen3-Embedding-0.6B"
        print(f"------select_model:{self.SELECT_MODEL}")
        self.model = LLM(model=self.SELECT_MODEL, task="embed")
        self.mask_token = "<mask>"   # mask token used during schema linking
        self.value_token = "<unk>"   # value token used during schema linking

        train_mask_questions = mask_question_with_schema_linking(self.train_json, mask_tag=self.mask_token, value_tag=self.value_token)
        train_embeddings = self.model.embed(train_mask_questions)
        self.train_embeddings = torch.tensor([o.outputs.embedding for o in train_embeddings])
        self.train_embeddings = F.normalize(self.train_embeddings, p=2, dim=1)

    def get_examples(self, target, num_example, cross_domain=False):
        target_mask_question = mask_question_with_schema_linking([target], mask_tag=self.mask_token, value_tag=self.value_token)
        target_embedding = self.model.embed(target_mask_question)
        target_embedding = torch.tensor([o.outputs.embedding for o in target_embedding])
        target_embedding = F.normalize(target_embedding, p=2, dim=1)

        # find the most similar question in train dataset
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = np.squeeze(cosine_similarity(target_embedding, self.train_embeddings)).tolist()
        pairs = [(similarity, index) for similarity, index in zip(similarities, range(len(similarities)))]
        print("-----------------cosine similarities-----------------")

        train_json = self.train_json
        pairs_sorted = sorted(pairs, key=lambda x: x[0], reverse=True)
        top_pairs = []
        for d, index in pairs_sorted:
            similar_db_id = train_json[index]["db_id"]
            if cross_domain and similar_db_id == target["db_id"]:
                continue
            # skeleton similarity
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
                if jaccard_similarity(train_json[index]["query_skeleton"], target["query_skeleton"]) >= self.threshold:
                    continue
                top_pairs.append((index, d))
                if len(top_pairs) >= num_example:
                    break

        return [train_json[index] for (index, d) in top_pairs]


class EuclideanDistanceQuestionMaskEmbedSelector(BasicExampleSelector):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data)

        self.SELECT_MODEL = "/models/hf-models/Qwen3-Embedding-0.6B"
        self.model = LLM(model=self.SELECT_MODEL, task="embed")
        self.mask_token = "<mask>"
        self.value_token = "<unk>"

        from sentence_transformers import SentenceTransformer
        train_mask_questions = mask_question_with_schema_linking(self.train_json, mask_tag=self.mask_token, value_tag=self.value_token)
        self.bert_model = SentenceTransformer(self.SELECT_MODEL, device="cpu")
        train_embeddings = self.model.embed(train_mask_questions)
        self.train_embeddings = torch.tensor([o.outputs.embedding for o in train_embeddings])
        self.train_embeddings = F.normalize(self.train_embeddings, p=2, dim=1)

    def get_examples(self, target, num_example, cross_domain=False):
        target_mask_question = mask_question_with_schema_linking([target], mask_tag=self.mask_token, value_tag=self.value_token)
        target_embedding = self.model.embed(target_mask_question)
        target_embedding = torch.tensor([o.outputs.embedding for o in target_embedding])
        target_embedding = F.normalize(target_embedding, p=2, dim=1)

        # find the most similar question in train dataset
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = np.squeeze(cosine_similarity(target_embedding, self.train_embeddings)).tolist()
        pairs = [(similarity, index) for similarity, index in zip(similarities, range(len(similarities)))]
        print("-----------------cosine similarities-----------------")

        pairs_sorted = sorted(pairs, key=lambda x: x[0], reverse=True)
        train_json = self.train_json
        top_pairs = []
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

        self.SELECT_MODEL = "/models/hf-models/all-mpnet-base-v2"
        self.mask_token = "<mask>"
        self.value_token = "<unk>"

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
        top_pairs = []
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
        self.SELECT_MODEL = "/models/hf-models/Qwen3-Embedding-0.6B"
        print(f"------select_model:{self.SELECT_MODEL}")
        self.model = LLM(model=self.SELECT_MODEL, task="embed")

        train_embeddings = self.model.embed(self.train_questions)
        self.train_embeddings = torch.tensor([o.outputs.embedding for o in train_embeddings])
        self.train_embeddings = F.normalize(self.train_embeddings, p=2, dim=1)

    def get_examples(self, target, num_example, cross_domain=False):
        target_embedding = self.model.embed([target["question"]])
        target_embedding = torch.tensor([o.outputs.embedding for o in target_embedding])
        target_embedding = F.normalize(target_embedding, p=2, dim=1)

        # find the most similar question in train dataset
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = np.squeeze(cosine_similarity(target_embedding, self.train_embeddings)).tolist()
        pairs = [(similarity, index) for similarity, index in zip(similarities, range(len(similarities)))]
        print("-----------------cosine similarities-----------------no mask")
        train_json = self.train_json
        pairs_sorted = sorted(pairs, key=lambda x: x[0], reverse=True)
        top_pairs = []
        for d, index in pairs_sorted:
            similar_db_id = train_json[index]["db_id"]
            if cross_domain and similar_db_id == target["db_id"]:
                continue
            # skeleton similarity
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
                if jaccard_similarity(train_json[index]["pre_skeleton"], target["pre_skeleton"]) >= self.threshold:
                    continue
                top_pairs.append((index, d))
                if len(top_pairs) >= num_example:
                    break

        return [train_json[index] for (index, d) in top_pairs]


class EuclideanDistancePreSkeletonSimilarPlusSelector(BasicExampleSelector):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data)

        self.SELECT_MODEL = "/models/hf-models/all-mpnet-base-v2"

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
        top_pairs = []
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
        self.SELECT_MODEL = "/models/hf-models/Qwen3-Embedding-0.6B"
        print(f"------select_model:{self.SELECT_MODEL}")
        self.model = LLM(model=self.SELECT_MODEL, task="embed")
        self.mask_token = "<mask>"
        self.value_token = "<unk>"

        train_mask_questions = mask_question_with_schema_linking(self.train_json, mask_tag=self.mask_token, value_tag=self.value_token)
        train_embeddings = self.model.embed(train_mask_questions)
        self.train_embeddings = torch.tensor([o.outputs.embedding for o in train_embeddings])
        self.train_embeddings = F.normalize(self.train_embeddings, p=2, dim=1)

    def get_examples(self, target, num_example, cross_domain=False):
        target_mask_question = mask_question_with_schema_linking([target], mask_tag=self.mask_token, value_tag=self.value_token)
        target_embedding = self.model.embed(target_mask_question)
        target_embedding = torch.tensor([o.outputs.embedding for o in target_embedding])
        target_embedding = F.normalize(target_embedding, p=2, dim=1)

        # find the most similar question in train dataset
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = np.squeeze(cosine_similarity(target_embedding, self.train_embeddings)).tolist()
        pairs = [(similarity, index) for similarity, index in zip(similarities, range(len(similarities)))]
        print("-----------------cosine similarities-----------------")
        
        train_json = self.train_json
        pairs_sorted = sorted(pairs, key=lambda x: x[0], reverse=True)
        top_pairs = []
        for d, index in pairs_sorted:
            similar_db_id = train_json[index]["db_id"]
            if cross_domain and similar_db_id == target["db_id"]:
                continue
            # skeleton similarity
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
                if jaccard_similarity(train_json[index]["pre_skeleton"], target["pre_skeleton"]) >= self.threshold:
                    continue
                top_pairs.append((index, d))
                if len(top_pairs) >= num_example:
                    break

        return [train_json[index] for (index, d) in top_pairs]


class EuclideanDistanceQuestionMaskPreSkeletonSimilarThresholdShiftSelector(BasicExampleSelector):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data)

        self.SELECT_MODEL = "/models/hf-models/all-mpnet-base-v2"
        self.mask_token = "<mask>"
        self.value_token = "<unk>"
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
        top_pairs = []
        for d, index in pairs_sorted:
            similar_db_id = train_json[index]["db_id"]
            if cross_domain and similar_db_id == target["db_id"]:
                continue
            # skeleton similarity
            if jaccard_similarity(train_json[index]["pre_skeleton"], target["pre_skeleton"]) < self.threshold:
                continue
            top_pairs.append((index, d))
            if len(top_pairs) >= num_example:
                break

        return [train_json[index] for (index, d) in top_pairs]


class EmbedSelector(BasicExampleSelector):
    def __init__(self, data, select_model_path, *args, **kwargs):
        super().__init__(data)

        self.SELECT_MODEL = select_model_path
        print(f"self.SELECT_MODEL:{self.SELECT_MODEL}")
        # e.g.:
        # /models/hf-models/Qwen3-Embedding-0.6B
        # /models/output/qwen3_emb_sft_0.6B_15/top5_random10/checkpoint-9636
        self.mask_token = "<mask>"
        self.value_token = "<unk>"

        from vllm import LLM
        self.model = LLM(model=self.SELECT_MODEL, task="embed")
        train_sqls = []
        for d in self.train_json:
            # if entry has 'query' field
            if 'query' in d:
                sql = d["query"]
            elif 'sql' in d:
                sql = d["sql"]
            else:
                raise KeyError("Data entry is missing a 'query' or 'sql' key:", d)
            train_sqls.append(sql)
        train_embeddings = self.model.embed(train_sqls)
        self.train_embeddings = torch.tensor([o.outputs.embedding for o in train_embeddings])
        self.train_embeddings = F.normalize(self.train_embeddings, p=2, dim=1)

    def get_examples(self, target, num_example, cross_domain=False):
        target_embedding = self.model.embed([target["question"]])
        target_embedding = torch.tensor([o.outputs.embedding for o in target_embedding])
        target_embedding = F.normalize(target_embedding, p=2, dim=1)
        train_embeddings = self.train_embeddings

        from sklearn.metrics.pairwise import cosine_similarity
        similarities = np.squeeze(cosine_similarity(target_embedding, self.train_embeddings)).tolist()
        pairs = [(similarity, index) for similarity, index in zip(similarities, range(len(similarities)))]
        pairs_sorted = sorted(pairs, key=lambda x: x[0], reverse=True)
        train_json = self.train_json
        top_pairs = []
        for d, index in pairs_sorted:
            similar_db_id = train_json[index]["db_id"]
            if cross_domain and similar_db_id == target["db_id"]:
                continue
            top_pairs.append((index, d))
            if len(top_pairs) >= num_example:
                break

        return [train_json[index] for (index, d) in top_pairs]
