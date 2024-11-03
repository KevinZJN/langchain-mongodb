from typing import AsyncGenerator, Generator

from ..utils import ConsistentFakeEmbeddings, MockCollection
from langchain_mongodb import MongoDBAtlasVectorSearch

import pytest
from langchain_core.vectorstores import VectorStore
from langchain_standard_tests.integration_tests.vectorstores import (
    AsyncReadWriteTestSuite,
    ReadWriteTestSuite,
)


class TestSync(ReadWriteTestSuite):
    @pytest.fixture()
    def vectorstore(self) -> Generator[VectorStore, None, None]:  # type: ignore
        """Get an empty vectorstore for unit tests."""
        return MongoDBAtlasVectorSearch(
            collection=MockCollection(), embeddings=ConsistentFakeEmbeddings()
        )