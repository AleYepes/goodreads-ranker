def _needs_embedding_model_migration(db_conn, table_name: str) -> bool:
    cursor = db_conn.execute(f"PRAGMA table_info({table_name})")
    columns = {row[1] for row in cursor.fetchall()}
    if not columns:
        return False
    return "embedding_model_id" not in columns and (
        "embedding_model" in columns or table_name in ("book_predictions", "prediction_hyperparams")
    )


def migrate_embedding_model_schema(db_conn):
    fk_row = db_conn.execute("PRAGMA foreign_keys").fetchone()
    original_fk = int(fk_row[0]) if fk_row else 1

    try:
        db_conn.execute("PRAGMA foreign_keys = OFF")
        db_conn.execute("BEGIN TRANSACTION")

        db_conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embedding_models (
                id   INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE
            )
            """
        )

        if _needs_embedding_model_migration(db_conn, "book_embeddings"):
            db_conn.execute("ALTER TABLE book_embeddings RENAME TO book_embeddings_old")
            db_conn.execute(
                """
                CREATE TABLE book_embeddings (
                    book_id             INTEGER REFERENCES books(legacy_id) ON DELETE CASCADE,
                    embedding_model_id  INTEGER NOT NULL REFERENCES embedding_models(id),
                    vector              BLOB NOT NULL,
                    text_hash           TEXT NOT NULL,
                    PRIMARY KEY (book_id, embedding_model_id)
                )
                """
            )
            db_conn.execute(
                "INSERT OR IGNORE INTO embedding_models (name) SELECT DISTINCT embedding_model FROM book_embeddings_old"
            )
            db_conn.execute(
                """
                INSERT INTO book_embeddings (book_id, embedding_model_id, vector, text_hash)
                SELECT o.book_id, m.id, o.vector, o.text_hash
                FROM book_embeddings_old o
                JOIN embedding_models m ON m.name = o.embedding_model
                """
            )
            db_conn.execute("DROP TABLE book_embeddings_old")
            print("✓ Migrated book_embeddings to embedding_model_id schema (data preserved).")

        if _needs_embedding_model_migration(db_conn, "prediction_hyperparams"):
            db_conn.execute("DROP TABLE IF EXISTS prediction_hyperparams")
            print("✓ Dropped prediction_hyperparams (old schema) — will repopulate on next predict run.")

        if _needs_embedding_model_migration(db_conn, "book_predictions"):
            db_conn.execute("DROP TABLE IF EXISTS book_predictions")
            print("✓ Dropped book_predictions (old schema) — will repopulate on next predict run.")

        db_conn.commit()
    except Exception:
        db_conn.rollback()
        raise
    finally:
        db_conn.execute(f"PRAGMA foreign_keys = {original_fk}")
