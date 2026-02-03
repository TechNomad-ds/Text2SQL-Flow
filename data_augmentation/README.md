Our data construction code has been fully integrated into the **DataFlow** project.

## 1. Setting Up the DataFlow Environment

First, clone the DataFlow repository and complete the local installation:

```shell
conda create -n text2sql_flow python=3.10
conda activate text2sql_flow
git clone https://github.com/OpenDCAI/DataFlow
cd DataFlow
pip install -e .
```

The steps above will automatically install all dependencies required by DataFlow and install it into the current Python environment in editable mode, which is convenient for further development and debugging.

---

## 2. Initializing the Working Directory

After installing DataFlow, return to the parent directory and create an experimental working directory:

```shell
cd ..
mkdir run_dataflow
cd run_dataflow
dataflow init
```

After initialization, a default pipeline file will be generated in the following path:

```
run_dataflow/api_pipelines/text2sql_pipeline_refine.py
```

The data construction workflow used in this project is implemented based on this pipeline file.

---

## 3. Configuring API Keys and Model Endpoints

### 3.1 Setting the API Key

Please set your API key via environment variables:

```shell
export DF_API_KEY="sk-xxxxx"
```

### 3.2 Configuring Model Services

In the pipeline code, configure the language model and embedding model used for data construction. For example:

```python
self.llm_serving = APILLMServing_request(
    api_url="https://api.openai.com/v1/chat/completions",  # Can be replaced with a custom endpoint
    model_name="gpt-4o",  # Project data was constructed using gpt-4o
    max_workers=100
)

self.embedding_serving = APILLMServing_request(
    api_url="https://api.openai.com/v1/embeddings",
    model_name="text-embedding-ada-002",  # Can be replaced with other embedding models
    max_workers=100
)
```

Where:

* **`llm_serving`**: The foundational large language model used for Text2SQL data construction.
* **`embedding_serving`**: Used to generate vector representations of candidate natural language queries.
  After generating multiple candidates, the optimal query is selected via vector similarity computation.

---

## 4. Database Configuration

In the `main` function of the pipeline, set the `db_root_path` parameter to the directory containing your databases.
In this project, SQLite databases are used.

An example directory structure is shown below:

```
databases/
  ├── california_schools/
  │   └── california_schools.sqlite
  └── hospitals/
      └── hospitals.sqlite
```

You can download the reference dataset from: [https://www.modelscope.cn/datasets/seeklhy/OmniSQL-datasets](https://www.modelscope.cn/datasets/seeklhy/OmniSQL-datasets)

---

## 5. Running the Pipeline

Simply run the pipeline script:

```shell
python run_dataflow/api_pipelines/text2sql_pipeline_refine.py
```

### Output and Caching

The output of each pipeline step will be saved in the `cache` directory.
This behavior is controlled by the `FileStorage` configuration:

```python
self.storage = FileStorage(
    first_entry_file_name="../example_data/Text2SQLPipeline/pipeline_refine.jsonl",  # Initial input file (can be replaced)
    cache_path="./cache",                      # Directory for cached outputs
    file_name_prefix="dataflow_cache_step",    # Cache file name prefix
    cache_type="jsonl"                         # File format
)
```

---

## 6. Code Structure Overview

All implementations related to this project are integrated into the DataFlow repository and organized as follows:

### Operator Implementations

```
DataFlow/dataflow/operators/text2sql
```

### Pipeline Implementation

```
DataFlow/dataflow/statics/pipelines/api_pipelines/text2sql_pipeline_refine.py
```

### Database Management and SQL Execution

```
DataFlow/dataflow/utils/text2sql
```

### Prompt Templates

```
DataFlow/dataflow/prompts/text2sql.py
```

---

## 7. Practical Considerations for Large-Scale Data Construction

In the DataFlow pipeline, some operators load all prompts into memory and then send them to the LLM for processing.
This may lead to out-of-memory issues when constructing large-scale datasets.
To address this, we provides a batch processing mechanism that reads and processes data in chunks.
We implemented this batching logic in the current directory and also provide a corresponding shell script to facilitate large-scale data construction.
