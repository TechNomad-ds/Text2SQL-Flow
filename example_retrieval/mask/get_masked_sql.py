import json
import os
import re
from sqlparse import parse as Parser
from tqdm import tqdm
from spider_match_utils import match_shift


import json
import os

from utils import get_tables, sql2skeleton

class BasicDataset(object):
    def __init__(self, path_data, pre_test_result=None):
        self.path_data = os.path.join(path_data, self.name)
        self.path_db = os.path.join(self.path_data, "database")
        self.test_json = os.path.join(self.path_data, self.test_json)
        self.test_gold = os.path.join(self.path_data, self.test_gold)
        self.train_json = os.path.join(self.path_data, self.train_json)
        self.train_gold = os.path.join(self.path_data, self.train_gold)
        self.table_json = os.path.join(self.path_data, self.table_json)
        self.path_test_schema_linking = os.path.join(self.path_data, "enc/test_schema-linking.jsonl")
        self.path_train_schema_linking = os.path.join(self.path_data, "enc/train_schema-linking_var.jsonl")
        if self.mini_test_index_json:
            self.mini_test_index_json = os.path.join(self.path_data, self.mini_test_index_json)
        else:
            self.mini_test_index_json = None

        self.pre_test_result = pre_test_result
            
        # lazy load for tables
        self.databases = None
        self.databases = self.get_databases()

    # test a mini set
    def set_mini_test(self, mini_file):
        self.mini_test_index_json = os.path.join(self.path_data, mini_file)

    def get_databases(self):
        if self.databases is None:
            self.databases = dict()
            # for db_id in os.listdir(self.path_db):
            #     self.databases[db_id] = self.get_tables(db_id)
            with open(self.table_json) as f:
                print("Loading table.json for db_id and tables ...")
                # print(f"self.table_json:{self.table_json}")
                tables = json.load(f)
                for tj in tables:
                    db_id = tj["db_id"]
                    self.databases[db_id] = self.get_tables(db_id)
        return self.databases

    def get_tables(self, db_id):
        if db_id in self.databases:
            return self.databases[db_id]
        else:
            path_db = os.path.join(self.path_db, db_id, db_id + ".sqlite")
            tables = get_tables(path_db)
            self.databases[db_id] = tables
            return tables

    def get_path_sql(self, db_id):
        path_sql = os.path.join(self.path_db, db_id, "schema.sql")
        return path_sql
        
    def get_table_json(self):
        return json.load(open(self.table_json, "r"))

    def get_path_db(self, db_id):
        return os.path.join(self.path_db, db_id, f"{db_id}.sqlite")

    def get_train_questions(self):
        questions = json.load(open(self.train_json, "r"))
        return [_["question"] for _ in questions]

    def get_mini_index(self):
        if self.mini_test_index_json:
            return json.load(open(self.mini_test_index_json, "r"))
        else:
            return None

    def get_test_questions(self, mini_set=False):
        questions = json.load(open(self.test_json, "r"))
        if mini_set and self.mini_test_index_json:
            mini_test_index = self.get_mini_index()
            questions = [questions[i] for i in mini_test_index]
        return [_["question"] for _ in questions]

    # get query skeletons
    def get_pre_skeleton(self, queries=None, schemas=None, mini_set=False):
        if queries:
            skeletons = []
            cnt = 0
            for query,schema in zip(queries, schemas):
                print("Processing query:", query)
                cnt += 1
                print(f"Processing query {cnt}/{len(queries)}")
                skeleton, masked_sql = sql2skeleton(query, schema)
                skeletons.append(skeleton)
            if mini_set and self.mini_test_index_json:
                mini_index = self.get_mini_index()
                skeletons = [skeletons[i] for i in mini_index]
            return skeletons
        else:
            return False

    # get all train information
    def get_train_json(self):
        print(f"self.train_json:{self.train_json}")
        datas = json.load(open(self.train_json, "r"))
        if datas is None:
            print(f"----------error")
        linking_infos = self.get_train_schema_linking()
        db_id_to_table_json = dict()
        for table_json in self.get_table_json():
            db_id_to_table_json[table_json["db_id"]] = table_json
        schemas = [db_id_to_table_json[d["db_id"]] for d in datas]
        queries = [data["query"] for data in datas]
        pre_queries = self.get_pre_skeleton(queries, schemas)
        return self.data_pre_process(datas, linking_infos, pre_queries)

    # get all test information
    def get_test_json(self, mini_set=False):
        tests = json.load(open(self.test_json, "r"))
        if mini_set and self.mini_test_index_json:
            mini_test_index = self.get_mini_index()
            tests = [tests[i] for i in mini_test_index]
        linking_infos = self.get_test_schema_linking(mini_set)
        db_id_to_table_json = dict()
        for table_json in self.get_table_json():
            db_id_to_table_json[table_json["db_id"]] = table_json
        schemas = [db_id_to_table_json[d["db_id"]] for d in tests]
        if self.pre_test_result:
            with open(self.pre_test_result, 'r') as f:
                lines = f.readlines()
                queries = [line.strip() for line in lines]
                pre_queries = self.get_pre_skeleton(queries, schemas, mini_set)
        else:
            pre_queries = None
        return self.data_pre_process(tests, linking_infos, pre_queries)

    def get_test_schema_linking(self, mini_set=False):
        if not os.path.exists(self.path_test_schema_linking):
            return None
        linking_infos = []
        with open(self.path_test_schema_linking, 'r') as f:
            for line in f.readlines():
                if line.strip():
                    linking_infos.append(json.loads(line))
        if mini_set and self.mini_test_index_json:
            mini_test_index = self.get_mini_index()
            linking_infos = [linking_infos[i] for i in mini_test_index]
        return linking_infos

    def get_train_schema_linking(self):
        if not os.path.exists(self.path_train_schema_linking):
            return None
        linking_infos = []
        print(f"path_train_schema_linking:{self.path_train_schema_linking}")
        cnt = 0
        with open(self.path_train_schema_linking, 'r') as f:
            for line in f.readlines():
                if line.strip():
                    try:
                        linking_infos.append(json.loads(line))
                        cnt = cnt + 1
                    except json.decoder.JSONDecodeError as e:
                        print(f"  ERROR: {e.msg}")
                        continue
        print(f"Analyzed {cnt} lines of training schema linking info.")
        return linking_infos

    def get_all_json(self):
        return self.get_train_json() + self.get_test_json()

    def get_train_answers(self):
        with open(self.train_gold, "r") as file:
            answers = file.readlines()
            return answers

    def get_test_answers(self, mini_set=False):
        with open(self.test_gold, "r") as file:
            answers = file.readlines()
            if mini_set and self.mini_test_index_json:
                mini_test_index = self.get_mini_index()
                answers = [answers[i] for i in mini_test_index]
            return answers

    def get_train_duplicated_index(self):
        train_data = self.get_train_json()
        example_dict = {}
        duplicated_index = []
        for i in range(len(train_data)):
            db_id = train_data[i]["db_id"]
            question = train_data[i]["question"]
            if (db_id, question) in example_dict.keys():
                duplicated_index.append(i)
            else:
                example_dict[(db_id, question)] = True
        return duplicated_index

    # get skeletons and schema_linking info
    def data_pre_process(self, datas, linking_infos=None, pre_queries=None):
        db_id_to_table_json = dict()
        for table_json in self.get_table_json():
            db_id_to_table_json[table_json["db_id"]] = table_json
        for data in datas:
            db_id = data["db_id"]
            data["tables"] = self.get_tables(db_id)
            if data["query"].strip()[:6] != 'SELECT':
                data["query_skeleton"] = data["query"]
            else:
                data["query_skeleton"], data["masked_sql"] = sql2skeleton(data["query"], db_id_to_table_json[db_id])
            data["path_db"] = self.get_path_db(db_id)
        if linking_infos:
            db_id_to_table_json = dict()
            for table_json in self.get_table_json():
                db_id_to_table_json[table_json["db_id"]] = table_json
            for id in range(min(len(datas), len(linking_infos))):
                datas[id]["sc_link"] = linking_infos[id]["sc_link"]
                datas[id]["cv_link"] = linking_infos[id]["cv_link"]
                datas[id]["question_for_copying"] = linking_infos[id]["question_for_copying"]
                datas[id]["column_to_table"] = linking_infos[id]["column_to_table"]
                db_id = datas[id]["db_id"]
                datas[id]["table_names_original"] = db_id_to_table_json[db_id]["table_names_original"]
            question_patterns = get_question_pattern_with_schema_linking(datas)
            for id in range(len(datas)):
                datas[id]["question_pattern"] = question_patterns[id]
            maked_questions = mask_question_with_schema_linking(datas, "<mask>", "<unk>")
            for id in range(len(datas)):
                datas[id]["masked_question"] = maked_questions[id]
        if pre_queries:
            for id in range(min(len(datas), len(pre_queries))):
                datas[id]["pre_skeleton"] = pre_queries[id]
        return datas


class SpiderDataset(BasicDataset):
    name = "spider"
    dev_json = "dev.json"
    dev_gold = "dev_gold.sql"
    test_json = "test.json"
    test_gold = "test_gold.sql"
    # train_json = "train_spider_and_others.json"
    train_json = "train_varient.json"
    train_gold = "train_gold.sql"
    table_json = "tables.json"
    test_table_json = "test_tables.json"
    mini_test_index_json = "mini_dev_index.json"


class RealisticDataset(BasicDataset):
    # only used for data path, shared with spider
    name = "spider_realistic"
    test_json = "spider-realistic.json"
    test_gold = "spider-realistic_gold.sql"
    train_json = "train_spider_and_others.json"
    train_gold = "train_gold.sql"
    table_json = "tables.json"
    mini_test_index_json = None

class BirdDataset(BasicDataset):
    name = "bird"
    test_json = "dev.json"
    test_gold = "dev.sql"
    # train_json = "train.json"
    train_json = "train_varient.json"
    train_gold = "train_gold.sql"
    table_json = "tables.json"
    mini_test_index_json = None


def load_data(data_type, path_data, pre_test_result=None):
    if data_type.lower() == "spider":
        return SpiderDataset(path_data, pre_test_result)
    elif data_type.lower() == "realistic":
        return RealisticDataset(path_data, pre_test_result)
    elif data_type.lower() == "bird":
        return BirdDataset(path_data, pre_test_result)
    else:
        raise RuntimeError()


def isNegativeInt(string):
    if string.startswith("-") and string[1:].isdigit():
        return True
    else:
        return False

def isFloat(string):
    if string.startswith("-"):
        string = string[1:]

    s = string.split(".")
    if len(s) > 2:
        return False
    else:
        for s_i in s:
            if not s_i.isdigit():
                return False
        return True

def sql_normalization(sql):
    sql = sql.strip()
    def white_space_fix(s):
        parsed_s = Parser(s)
        s = " ".join([token.value for token in parsed_s.tokens])

        return s

    # convert everything except text between single quotation marks to lower case
    def lower(s):
        in_quotation = False
        out_s = ""
        for char in s:
            if in_quotation:
                out_s += char
            else:
                out_s += char.lower()

            if char == "'":
                if in_quotation:
                    in_quotation = False
                else:
                    in_quotation = True

        return out_s

    # remove ";"
    def remove_semicolon(s):
        if s.endswith(";"):
            s = s[:-1]
        return s

    # double quotation -> single quotation
    def double2single(s):
        return s.replace("\"", "'")

    def add_asc(s):
        pattern = re.compile(r'order by (?:\w+ \( \S+ \)|\w+\.\w+|\w+)(?: (?:\+|\-|\<|\<\=|\>|\>\=) (?:\w+ \( \S+ \)|\w+\.\w+|\w+))*')
        if "order by" in s and "asc" not in s and "desc" not in s:
            for p_str in pattern.findall(s):
                s = s.replace(p_str, p_str + " asc")

        return s

    def sql_split(s):
        while "  " in s:
            s = s.replace("  ", " ")
        s = s.strip()
        i = 0
        toks = []
        while i < len(s):
            tok = ""
            if s[i] == "'":
                tok = tok + s[i]
                i += 1
                while i < len(s) and s[i] != "'":
                    tok = tok + s[i]
                    i += 1
                if i < len(s):
                    tok = tok + s[i]
                    i += 1
            else:
                while i < len(s) and s[i] != " ":
                    tok = tok + s[i]
                    i += 1
                while i < len(s) and s[i] == " ":
                    i += 1
            toks.append(tok)
        return toks

    def remove_table_alias(s):
        tables_aliases = Parser(s).tables_aliases
        new_tables_aliases = {}
        for i in range(1, 11):
            if "t{}".format(i) in tables_aliases.keys():
                new_tables_aliases["t{}".format(i)] = tables_aliases["t{}".format(i)]
        table_names = []
        for tok in sql_split(s):
            if '.' in tok:
                table_names.append(tok.split('.')[0])
        for table_name in table_names:
            if table_name in tables_aliases.keys():
                new_tables_aliases[table_name] = tables_aliases[table_name]
        tables_aliases = new_tables_aliases

        new_s = []
        pre_tok = ""
        for tok in sql_split(s):
            if tok in tables_aliases.keys():
                if pre_tok == 'as':
                    new_s = new_s[:-1]
                elif pre_tok != tables_aliases[tok]:
                    new_s.append(tables_aliases[tok])
            elif '.' in tok:
                split_toks = tok.split('.')
                for i in range(len(split_toks)):
                    if len(split_toks[i]) > 2 and split_toks[i][0] == "'" and split_toks[i][-1] == "'":
                        split_toks[i] = split_toks[i].replace("'", "")
                        split_toks[i] = split_toks[i].lower()
                    if split_toks[i] in tables_aliases.keys():
                        split_toks[i] = tables_aliases[split_toks[i]]
                new_s.append('.'.join(split_toks))
            else:
                new_s.append(tok)
            pre_tok = tok

        # remove as
        s = new_s
        new_s = []
        for i in range(len(s)):
            if s[i] == "as":
                continue
            if i > 0 and s[i-1] == "as":
                continue
            new_s.append(s[i])
        new_s = ' '.join(new_s)

        # for k, v in tables_aliases.items():
        #     s = s.replace("as " + k + " ", "")
        #     s = s.replace(k, v)

        return new_s

    processing_func = lambda x: remove_table_alias(add_asc(lower(white_space_fix(double2single(remove_semicolon(x))))))

    return processing_func(sql.strip())


# def sql2skeleton(sql: str, db_schema):
#     print(f"sql:{sql}")
#     sql = sql_normalization(sql)

#     table_names_original, table_dot_column_names_original, column_names_original = [], [], []
#     column_names_original.append("*")
#     for table_id, table_name_original in enumerate(db_schema["table_names_original"]):
#         table_names_original.append(table_name_original.lower())
#         table_dot_column_names_original.append(table_name_original + ".*")
#         for column_id_and_name in db_schema["column_names_original"]:
#             column_id = column_id_and_name[0]
#             column_name_original = column_id_and_name[1]
#             table_dot_column_names_original.append(table_name_original.lower() + "." + column_name_original.lower())
#             column_names_original.append(column_name_original.lower())

#     # print(f"sql:{sql}")
#     parsed_sql = Parser(sql)
#     # print(f"parsed_sql:{parsed_sql is not None}")
#     new_sql_tokens = []
#     for token in parsed_sql.tokens:
#         # mask table names
#         if token.value in table_names_original:
#             new_sql_tokens.append("<mask>")
#         # mask column names
#         elif token.value in column_names_original \
#                 or token.value in table_dot_column_names_original:
#             new_sql_tokens.append("<mask>")
#         # mask string values
#         elif token.value.startswith("'") and token.value.endswith("'"):
#             new_sql_tokens.append("<unk>")
#         elif token.value.startswith("`") and token.value.endswith("`"):
#             new_sql_tokens.append("<unk>")
#         elif token.value.startswith('"') and token.value.endswith('"'):
#             new_sql_tokens.append("<unk>")
#         # mask positive int number
#         elif token.value.isdigit():
#             new_sql_tokens.append("<unk>")
#         # mask negative int number
#         elif isNegativeInt(token.value):
#             new_sql_tokens.append("<unk>")
#         # mask float number
#         elif isFloat(token.value):
#             new_sql_tokens.append("<unk>")
#         else:
#             new_sql_tokens.append(token.value.strip())

#     sql_skeleton = " ".join(new_sql_tokens)
#     masked_sql = sql_skeleton

#     # remove JOIN ON keywords
#     sql_skeleton = sql_skeleton.replace("on _ = _ and _ = _", "on _ = _")
#     sql_skeleton = sql_skeleton.replace("on _ = _ or _ = _", "on _ = _")
#     sql_skeleton = sql_skeleton.replace(" on _ = _", "")
#     pattern3 = re.compile("_ (?:join _ ?)+")
#     sql_skeleton = re.sub(pattern3, "_ ", sql_skeleton)

#     # "_ , _ , ..., _" -> "_"
#     while ("_ , _" in sql_skeleton):
#         sql_skeleton = sql_skeleton.replace("_ , _", "_")

#     # remove clauses in WHERE keywords
#     ops = ["=", "!=", ">", ">=", "<", "<="]
#     for op in ops:
#         if "_ {} _".format(op) in sql_skeleton:
#             sql_skeleton = sql_skeleton.replace("_ {} _".format(op), "_")
#     while ("where _ and _" in sql_skeleton or "where _ or _" in sql_skeleton):
#         if "where _ and _" in sql_skeleton:
#             sql_skeleton = sql_skeleton.replace("where _ and _", "where _")
#         if "where _ or _" in sql_skeleton:
#             sql_skeleton = sql_skeleton.replace("where _ or _", "where _")

#     # remove additional spaces in the skeleton
#     while "  " in sql_skeleton:
#         sql_skeleton = sql_skeleton.replace("  ", " ")

#     # double check for order by
#     split_skeleton = sql_skeleton.split(" ")
#     for i in range(2, len(split_skeleton)):
#         if split_skeleton[i-2] == "order" and split_skeleton[i-1] == "by" and split_skeleton[i] != "_":
#             split_skeleton[i] = "_"
#     sql_skeleton = " ".join(split_skeleton)

#     return sql_skeleton, masked_sql



def mask_question_with_schema_linking(data_jsons, mask_tag, value_tag):
    mask_questions = []
    for data_json in data_jsons:
        sc_link = data_json["sc_link"]
        cv_link = data_json["cv_link"]
        q_col_match = sc_link["q_col_match"]
        q_tab_match = sc_link["q_tab_match"]
        num_date_match = cv_link["num_date_match"]
        cell_match = cv_link["cell_match"]
        question_for_copying = data_json["question_for_copying"]
        q_col_match, q_tab_match, cell_match = match_shift(q_col_match, q_tab_match, cell_match)

        def mask(question_toks, mask_ids, tag):
            new_question_toks = []
            for id, tok in enumerate(question_toks):
                if id in mask_ids:
                    new_question_toks.append(tag)
                else:
                    new_question_toks.append(tok)
            return new_question_toks

        num_date_match_ids = [int(match.split(',')[0]) for match in num_date_match]
        cell_match_ids = [int(match.split(',')[0]) for match in cell_match]
        value_match_q_ids = num_date_match_ids + cell_match_ids
        question_toks = mask(question_for_copying, value_match_q_ids, value_tag)

        q_col_match_ids = [int(match.split(',')[0]) for match in q_col_match]
        q_tab_match_ids = [int(match.split(',')[0]) for match in q_tab_match]
        schema_match_q_ids = q_col_match_ids + q_tab_match_ids
        question_toks = mask(question_toks, schema_match_q_ids, mask_tag)
        mask_questions.append(" ".join(question_toks))

    return mask_questions


def get_question_pattern_with_schema_linking(data_jsons):
    question_patterns = []
    for data_json in data_jsons:
        print(f"data_json['question']: {data_json['question']}")
        sc_link = data_json["sc_link"]
        cv_link = data_json["cv_link"]
        q_col_match = sc_link["q_col_match"]
        q_tab_match = sc_link["q_tab_match"]
        num_date_match = cv_link["num_date_match"]
        cell_match = cv_link["cell_match"]
        question_for_copying = data_json["question_for_copying"]

        def mask(question_toks, mask_ids, tag):
            new_question_toks = []
            for id, tok in enumerate(question_toks):
                if id in mask_ids:
                    new_question_toks.append(tag)
                else:
                    new_question_toks.append(tok)
            return new_question_toks

        num_date_match_ids = [int(match.split(',')[0]) for match in num_date_match]
        cell_match_ids = [int(match.split(',')[0]) for match in cell_match]
        value_match_q_ids = num_date_match_ids + cell_match_ids
        question_toks = mask(question_for_copying, value_match_q_ids, '_')

        q_col_match_ids = [int(match.split(',')[0]) for match in q_col_match]
        q_tab_match_ids = [int(match.split(',')[0]) for match in q_tab_match]
        schema_match_q_ids = q_col_match_ids + q_tab_match_ids
        question_toks = mask(question_toks, schema_match_q_ids, '_')
        question_patterns.append(" ".join(question_toks))

    return question_patterns


if __name__ == "__main__":
    train_json = "train_augmented.json"

    with open(train_json, 'r') as f:
        train_data = json.load(f)
    print(f"len(train_data):{len(train_data)}")
    PATH_DATA = "/mnt/public/gpfs-jd/data/lh/xc/code/DAIL-SQL/dataset/"
    data = load_data("spider", PATH_DATA)
    train_data_masked = data.get_train_json()
    print(f"len(train_data_masked):{len(train_data_masked)}")
    assert len(train_data) == len(train_data_masked)
    for i in range(len(train_data)):
        assert train_data[i]['query'] == train_data_masked[i]['query']
        assert train_data[i]['question'] == train_data_masked[i]['question']
        train_data[i]['masked_question'] = train_data_masked[i]['masked_question']
        train_data[i]['masked_sql'] = train_data_masked[i]['masked_sql']
    with open("train_augmented_masked.json", 'w') as f:
        json.dump(train_data, f, indent=4)


