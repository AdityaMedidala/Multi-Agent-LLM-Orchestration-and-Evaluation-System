"""add_document_chunks

Revision ID: 2e72f443f2ca
Revises: 6c993add8af3
Create Date: 2026-05-08 13:16:55.459407

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '2e72f443f2ca'
down_revision: Union[str, Sequence[str], None] = '6c993add8af3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    op.execute("""
        CREATE TABLE document_chunks (
            chunk_id    TEXT PRIMARY KEY,
            text        TEXT NOT NULL,
            metadata    JSONB,
            embedding   vector(1536),
            ts          TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', coalesce(text, ''))) STORED
        )
    """)
    op.execute("CREATE INDEX ON document_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)")
    op.execute("CREATE INDEX ON document_chunks USING GIN (ts)")

def downgrade():
    op.drop_table("document_chunks")