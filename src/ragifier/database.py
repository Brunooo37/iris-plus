import random

import duckdb
import lancedb

from ragifier.config import DatabaseConfig

# TODO table.optimize()
# TODO table.create_index(accelerator="cuda")


# TODO implement
def make_batches():
    pass


def make_database(cfg: DatabaseConfig):
    db = lancedb.connect(cfg.database_path)
    # schema is not required because polars already has a schema
    table = db.create_table(cfg.table_name, data=make_batches, mode="overwrite")

    table = db.open_table(cfg.table_name)
    _ = table.to_lance()  # type: ignore
    # this is black magic
    df = duckdb.sql("SELECT * FROM _ LIMIT 10").to_df()
    print(df)

    # fast filtering using an index
    table.create_scalar_index("id", index_type="BTREE")
    # FIXME these will be a batch of sample indices
    indices = ",".join([str(random.randint(0, 10)) for x in range(10)])
    df = table.search().where(f"id in ({indices})").to_polars()
    print(df)
