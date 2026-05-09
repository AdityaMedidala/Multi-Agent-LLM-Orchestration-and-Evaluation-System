"""add document_chunks table

Revision ID: 4a2bff979df8
Revises: 2e72f443f2ca
Create Date: 2026-05-09 05:14:12.514989

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '4a2bff979df8'
down_revision: Union[str, Sequence[str], None] = '2e72f443f2ca'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add created_at — the only column missing from the original raw-SQL table
    op.add_column(
        'document_chunks',
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
    )

    # Swap ivfflat index for HNSW (better recall, no training step required)
    op.execute(
        "DROP INDEX IF EXISTS document_chunks_embedding_idx"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_chunks_vector "
        "ON document_chunks USING hnsw (embedding vector_cosine_ops)"
    )

    # Ensure named GIN index exists for BM25 queries
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_chunks_ts "
        "ON document_chunks USING GIN (ts)"
    )


def downgrade() -> None:
    op.drop_column('document_chunks', 'created_at')
    op.execute("DROP INDEX IF EXISTS idx_chunks_vector")
    op.execute(
        "CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx "
        "ON document_chunks USING ivfflat (embedding vector_cosine_ops) "
        "WITH (lists = 100)"
    )
