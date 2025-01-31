import lancedb

from ragifier.config import DatabaseConfig


def make_batches():
    pass


def make_database(cfg: DatabaseConfig):
    db = lancedb.connect(cfg.database_path)
    db.create_table(cfg.table_name, data=make_batches, mode="overwrite")
