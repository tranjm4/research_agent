"""
Unit tests for the Embedder class using pytest framework
"""

import json
import time
import pytest
from unittest.mock import Mock, MagicMock, patch, call
from openai import RateLimitError
import os

from typing import Literal, Dict
from typing_extensions import TypedDict

from server.vectorstore.build.embed.embed import Embedder, VSDocument, VSEmbedding



class TestEmbedder:
    
    @pytest.fixture
    def mock_dependencies(self):
        with patch("server.vectorstore.build.embed.embed.KafkaConsumer") as mock_consumer, \
             patch("server.vectorstore.build.embed.embed.OpenAI") as mock_openai, \
             patch("server.vectorstore.build.embed.embed.MongoClient") as mock_mongo:
            yield mock_consumer, mock_openai, mock_mongo
            
    @pytest.fixture
    def mock_environment_variables(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_api_key", "BATCH_SIZE": "100"}):
            yield

    def test_embedder_initialization(self, mock_dependencies, mock_environment_variables):
        """Test the initialization of the Embedder class."""
        embedder = Embedder()
        assert embedder.embedding_model == "text-embedding-3-small"
        assert embedder.openai_client is not None
        assert embedder.consumer is not None
        assert embedder.db is not None

    def test_environment_variables_used_in_embedder_init(self, mock_dependencies, mock_environment_variables):
        """Test the environment variables used in Embedder initialization."""
        mock_consumer, mock_openai, mock_mongo = mock_dependencies
        embedder = Embedder()
        
        mock_openai.assert_called_once_with(api_key="test_api_key")

        assert embedder.batch_size == 100
        
    def test_run_process_single_batch(self, mock_dependencies, mock_environment_variables):
        """Test the run method with an even batch size."""
        embedder = Embedder()
        embedder.consumer = MagicMock()
        embedder.consumer.poll.side_effect = [
            {'key': [MagicMock(value='{"content": "test", "metadata": {"paper_id": "1"}}') for _ in range(100)]},
            {}
        ]
        embedder.process_batch_job = MagicMock()
        embedder.run()

        embedder.process_batch_job.assert_called_once()
    
    def test_run_process_residual_batch(self, mock_dependencies, mock_environment_variables):
        """Test the run method with batch size + some residual messages."""
        embedder = Embedder()
        embedder.consumer = MagicMock()
        embedder.consumer.poll.side_effect = [
            {'key': [MagicMock(value='{"content": "test", "metadata": {"paper_id": "1"}}') for _ in range(105)]},
            {}
        ]
        embedder.process_batch_job = MagicMock()
        embedder.run()

        assert embedder.process_batch_job.call_count == 2
        
    def test_run_process_multiple_polls_sum_to_batch(self, mock_dependencies, mock_environment_variables):
        """Test the run method with multiple batch sizes adding to a full batch."""
        embedder = Embedder()
        embedder.consumer = MagicMock()
        embedder.consumer.poll.side_effect = [
            {'key': [MagicMock(value='{"content": "test", "metadata": {"paper_id": "1"}}') for _ in range(50)]},
            {'key': [MagicMock(value='{"content": "test", "metadata": {"paper_id": "2"}}') for _ in range(50)]},
            {}
        ]
        embedder.process_batch_job = MagicMock()
        embedder.run()

        assert embedder.process_batch_job.call_count == 1
        
    def test_run_process_multiple_polls_sum_above_batch(self, mock_dependencies, mock_environment_variables):
        """Test the run method with multiple batch sizes exceeding a full batch."""
        embedder = Embedder()
        embedder.consumer = MagicMock()
        embedder.consumer.poll.side_effect = [
            {'key': [MagicMock(value='{"content": "test", "metadata": {"paper_id": "1"}}') for _ in range(50)]},
            {'key': [MagicMock(value='{"content": "test", "metadata": {"paper_id": "2"}}') for _ in range(60)]},
            {}
        ]
        embedder.process_batch_job = MagicMock()
        embedder.run()

        assert embedder.process_batch_job.call_count == 2
        
    def test_run_process_with_empty_poll_in_between_calls(self, mock_dependencies, mock_environment_variables):
        """Tests the run method with an empty poll in between poll calls"""
        embedder = Embedder()
        embedder.consumer = MagicMock()
        embedder.consumer.poll.side_effect = [
            {'key': [MagicMock(value='{"content": "test", "metadata": {"paper_id": "1"}}') for _ in range(50)]},
            {},
            {'key': [MagicMock(value='{"content": "test", "metadata": {"paper_id": "2"}}') for _ in range(60)]},
        ]
        embedder.process_batch_job = MagicMock()
        embedder.run()

        assert embedder.process_batch_job.call_count == 2

    def test_run_process_with_empty_poll_in_between_calls(self, mock_dependencies, mock_environment_variables):
        """Tests the run method with an empty poll in between poll calls"""
        embedder = Embedder()
        embedder.consumer = MagicMock()
        embedder.consumer.poll.side_effect = [
            {'key': [MagicMock(value='{"content": "test", "metadata": {"paper_id": "1"}}') for _ in range(50)]},
            {'key': []},
            {'key': [MagicMock(value='{"content": "test", "metadata": {"paper_id": "2"}}') for _ in range(60)]},
            {}
        ]
        embedder.process_batch_job = MagicMock()
        embedder.run()

        assert embedder.process_batch_job.call_count == 2

    def test_create_batch_file_format(self, mock_dependencies, mock_environment_variables):
        """Tests the create_batch_file method for correct file format."""
        embedder = Embedder()
        
        # Create the mock batch to process (1 file)
        mock_documents = [{
            "content": "This is a test document.",
            "metadata": {"paper_id": "123"}
        }]
        
        written_data = []
        
        def capture_json_dump(data, file):
            written_data.append(data)
        
        mock_file = MagicMock()
        mock_file.name = "/tmp/test_batch.jsonl"
        
        with patch("tempfile.NamedTemporaryFile", return_value=mock_file), \
             patch("json.dump", side_effect=capture_json_dump), \
             patch("builtins.open", MagicMock()), \
             patch("os.unlink"), \
             patch.object(embedder.openai_client.files, "create") as mock_create, \
             patch.object(embedder.db["embeddings_metadata"], "insert_many"):
            
            mock_create.return_value.id = "file-123"
            
            result = embedder.create_batch_file(mock_documents)
            
            # Verify the batch request format
            assert len(written_data) == 1
            request = written_data[0]
            
            assert request["method"] == "POST"
            assert request["url"] == "/v1/embeddings"
            assert request["custom_id"] == "batch_doc_0_123"
            assert request["body"]["model"] == embedder.embedding_model
            assert request["body"]["input"] == "This is a test document."
            assert request["body"]["encoding_format"] == "float"
            assert result == "file-123"

    def test_embedder_submits_batch_job(self, mock_dependencies, mock_environment_variables):
        """Tests the submit_batch_job method for correct behavior."""
        _, mock_openai, _ = mock_dependencies
        embedder = Embedder()
        embedder.openai_client = mock_openai
        embedder.create_batch_file = MagicMock(return_value="file-123")
        embedder.process_batch_job = MagicMock()

        embedder.submit_batch_job("batch-123")
        mock_openai.batches.create.assert_called_once_with(
            input_file_id="batch-123",
            endpoint="/v1/embeddings",
            completion_window="24h"
        )

    def test_process_batch_job(self, mock_dependencies, mock_environment_variables):
        """Tests the process_batch_job method for correct behavior."""
        embedder = Embedder()
        with patch.object(embedder, "create_batch_file") as mock_create_batch_file, \
            patch.object(embedder, "submit_batch_job") as mock_submit_batch_job:

            mock_create_batch_file.return_value = "file-123"
            mock_submit_batch_job.return_value = MagicMock(id="batch-123")

            embedder.process_batch_job("batch-123")
            
            mock_create_batch_file.assert_called_once_with("batch-123")

    def test_mongo_failure(self, mock_dependencies, mock_environment_variables):
        """Tests for failure case with MongoDB connection"""
        _, _, mock_mongo = mock_dependencies
        mock_mongo.side_effect = Exception("Failed to connect to MongoDB")

        with pytest.raises(Exception):
            embedder = Embedder()
            
