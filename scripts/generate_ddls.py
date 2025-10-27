import json
import sqlite3
import os

def obtain_db_schema(db_file_dir):
    conn = sqlite3.connect(db_file_dir)
    cursor = conn.cursor()

    cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    table_names = []
    create_statements = []
    for table in tables:
        table_name, create_statement = table
        table_names.append(table_name)
        create_statements.append(create_statement)

    cursor.close()
    conn.close()

    return table_names, create_statements


db_base_path = ""
tables = json.load(open(""))
for table in tables:
    db_path = os.path.join(db_base_path, table["db_id"], table["db_id"] + ".sqlite")
    table_names, create_statements = obtain_db_schema(db_path)
    table["ddls"] = create_statements

json.dump(tables, open("", "w"), indent=2, ensure_ascii=False)