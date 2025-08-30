import pytest
import os
from unittest.mock import patch, MagicMock
from server.vectorstore.build.loader import DBLoader

class TestDBLoader:
    
    @patch("server.vectorstore.build.loader.KAFKA_TOPIC_DOCS", "test_topic")
    @patch("server.vectorstore.build.loader.KafkaProducer")
    @patch("server.vectorstore.build.loader.MongoClient")
    @patch("server.vectorstore.build.loader.KAFKA_BOOTSTRAP_SERVERS", "localhost:9092,localhost:9093")
    @patch.dict(os.environ, {'MONGO_DB_NAME': 'test_db'})
    def test_db_loader_init_with_env_vars(self, mock_mongo_client, mock_kafka_producer):
        """Test DBLoader initialization with mocked environment variables"""
        mock_client_instance = MagicMock()
        mock_mongo_client.return_value = mock_client_instance
        mock_client_instance.__getitem__.return_value = MagicMock()
        
        mock_producer_instance = MagicMock()
        mock_kafka_producer.return_value = mock_producer_instance
        
        db_client_uri = "mongodb://localhost:27017"
        db_loader = DBLoader(db_uri=db_client_uri)
        
        assert db_loader.uri == db_client_uri
        mock_kafka_producer.assert_called_once_with(bootstrap_servers=['localhost:9092', 'localhost:9093'],
                                                    batch_size=16384,
                                                    linger_ms=100,
                                                    buffer_memory=67108864,
                                                    max_block_ms=5000,
                                                    retries=3,
                                                    acks=1)
        mock_mongo_client.assert_called_once_with(db_client_uri)

    @patch("server.vectorstore.build.loader.KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    @patch("server.vectorstore.build.loader.MongoClient")
    @patch.dict(os.environ, {'MONGO_DB_NAME': 'test_db'})
    def test_connect_to_db_success(self, mock_mongo_client):
        """Test successful database connection"""
        mock_client_instance = MagicMock()
        mock_mongo_client.return_value = mock_client_instance
        mock_client_instance.admin.command.return_value = True
        mock_db = MagicMock()
        mock_client_instance.__getitem__.return_value = mock_db
        
        with patch("server.vectorstore.build.loader.KafkaProducer"):
            db_loader = DBLoader(db_uri="mongodb://localhost:27017")
            
            mock_client_instance.admin.command.assert_called_once_with("ping")
            assert db_loader.db == mock_db

    @patch("server.vectorstore.build.loader.KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    @patch("server.vectorstore.build.loader.MongoClient")
    @patch.dict(os.environ, {'MONGO_DB_NAME': 'test_db'})
    def test_connect_to_db_failure(self, mock_mongo_client):
        """Test database connection failure handling"""
        from pymongo.errors import ConnectionFailure
        
        mock_client_instance = MagicMock()
        mock_mongo_client.return_value = mock_client_instance
        mock_client_instance.admin.command.side_effect = ConnectionFailure("Connection failed")
        mock_db = MagicMock()
        mock_client_instance.__getitem__.return_value = mock_db
        
        with patch("server.vectorstore.build.loader.KafkaProducer"), \
             patch("builtins.print") as mock_print:
            db_loader = DBLoader(db_uri="mongodb://localhost:27017")
            
            mock_print.assert_called_with("Failed to connect to MongoDB")
            assert db_loader.db == mock_db

    @patch("server.vectorstore.build.loader.KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    @patch("server.vectorstore.build.loader.KAFKA_TOPIC_DOCS", "test_topic")
    @patch("server.vectorstore.build.loader.KafkaProducer")
    @patch("server.vectorstore.build.loader.MongoClient")
    @patch.dict(os.environ, {'MONGO_DB_NAME': 'test_db'})
    def test_send_message_batch(self, mock_mongo_client, mock_kafka_producer):
        """Test sending batch messages to Kafka"""
        mock_client_instance = MagicMock()
        mock_mongo_client.return_value = mock_client_instance
        mock_client_instance.__getitem__.return_value = MagicMock()
        
        db_loader = DBLoader(db_uri="mongodb://localhost:27017")
        db_loader.producer = MagicMock()
        mock_producer_instance = db_loader.producer

        test_messages = [
            {"chunk_text": "message 1", "paper_id": "123", "title": "Test Paper 1"},
            {"chunk_text": "message 2", "paper_id": "456", "title": "Test Paper 2"}
        ]
        progress = (1, 2)
        db_loader.send_message_batch(test_messages, progress)
        
        assert mock_producer_instance.send.call_count == 2
        # Verify that messages were sent to the correct topic
        call_args = mock_producer_instance.send.call_args_list
        assert all(call[0][0] == 'test_topic' for call in call_args)
        # Verify the transformed queue documents contain expected content
        sent_values = [call[1]['value'].decode('utf-8') for call in call_args]
        assert "'content': 'message 1'" in sent_values[0]
        assert "'content': 'message 2'" in sent_values[1]

    @patch("server.vectorstore.build.loader.KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    @patch("server.vectorstore.build.loader.KAFKA_TOPIC_DOCS", "test_topic")
    @patch("server.vectorstore.build.loader.KafkaProducer")
    @patch("server.vectorstore.build.loader.MongoClient")
    @patch.dict(os.environ, {'MONGO_DB_NAME': 'test_db'})
    def test_send_message_batch_error(self, mock_mongo_client, mock_kafka_producer):
        """Test error handling in send_message_batch"""
        mock_producer_instance = MagicMock()
        mock_producer_instance.send.side_effect = Exception("Kafka error")
        mock_kafka_producer.return_value = mock_producer_instance
        
        mock_client_instance = MagicMock()
        mock_mongo_client.return_value = mock_client_instance
        mock_client_instance.__getitem__.return_value = MagicMock()
        
        with patch("server.vectorstore.build.loader.logger") as mock_logger:
            db_loader = DBLoader(db_uri="mongodb://localhost:27017")
            test_messages = [{"id": 1, "content": "message 1"}]
            
            progress = MagicMock(spec=tuple)
            db_loader.send_message_batch(test_messages, progress)

            mock_logger.error.assert_called_once()
    @patch("server.vectorstore.build.loader.KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    @patch("server.vectorstore.build.loader.KAFKA_TOPIC_DOCS", "test_topic")
    @patch("server.vectorstore.build.loader.tqdm")
    @patch("server.vectorstore.build.loader.KafkaProducer")
    @patch("server.vectorstore.build.loader.MongoClient")
    @patch.dict(os.environ, {'MONGO_DB_NAME': 'test_db'})
    def test_load_documents_batch_sends(self, mock_mongo_client, mock_kafka_producer, mock_tqdm):
        """Test the number of batch calls to send_message_batch"""
        # Mock Kafka producer
        mock_producer_instance = MagicMock()
        mock_kafka_producer.return_value = mock_producer_instance
        
        # Mock MongoDB
        mock_client_instance = MagicMock()
        mock_mongo_client.return_value = mock_client_instance
        mock_db = MagicMock()
        mock_client_instance.__getitem__.return_value = mock_db
        
        # Mock collection
        mock_collection = MagicMock()
        mock_db.__getitem__.return_value = mock_collection
        
        # Mock documents
        test_docs = [{"_id": i, "content": f"doc {i}"} for i in range(1050)]  # Test batch logic
        mock_collection.find.return_value = test_docs
        mock_collection.count_documents.return_value = len(test_docs)
        
        # Mock tqdm
        mock_tqdm.return_value = test_docs
        
        with patch("server.vectorstore.build.loader.logger") as mock_logger:
            db_loader = DBLoader(db_uri="mongodb://localhost:27017")
            db_loader.send_message_batch = MagicMock()
            db_loader.load_documents("test_collection")
            
            # Should send 2 batches (1000 + 50 docs)
            assert db_loader.send_message_batch.call_count == 2
            mock_logger.info.assert_called_with("Finished loading documents into Kafka.")
    
    @patch("server.vectorstore.build.loader.KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    @patch("server.vectorstore.build.loader.KAFKA_TOPIC_DOCS", "test_topic")
    @patch("server.vectorstore.build.loader.tqdm")
    @patch("server.vectorstore.build.loader.KafkaProducer")
    @patch("server.vectorstore.build.loader.MongoClient")
    @patch.dict(os.environ, {'MONGO_DB_NAME': 'test_db'})
    def test_load_documents_single_batch(self, mock_mongo_client, mock_kafka_producer, mock_tqdm):
        """Test loading documents with single batch (< 100 docs)"""
        # Mock Kafka producer
        mock_producer_instance = MagicMock()
        mock_kafka_producer.return_value = mock_producer_instance

        # Mock MongoDB
        mock_client_instance = MagicMock()
        mock_mongo_client.return_value = mock_client_instance
        mock_db = MagicMock()
        mock_client_instance.__getitem__.return_value = mock_db

        # Mock collection
        mock_collection = MagicMock()
        mock_db.__getitem__.return_value = mock_collection

        # Mock documents - only 50 docs (single batch)
        test_docs = [{"_id": i, "content": f"doc {i}"} for i in range(50)]
        mock_collection.find.return_value = test_docs
        mock_collection.count_documents.return_value = len(test_docs)

        # Mock tqdm
        mock_tqdm.return_value = test_docs

        with patch("server.vectorstore.build.loader.logger") as mock_logger:
            db_loader = DBLoader(db_uri="mongodb://localhost:27017")
            
            # Call load_documents with collection name (string), not list
            db_loader.load_documents("test_collection")

            # Should send 50 individual messages (one batch of 50)
            assert mock_producer_instance.send.call_count == 50
            mock_logger.info.assert_called_with("Finished loading documents into Kafka.")

    def test_missing_env_vars(self):
        """Test behavior when environment variables are missing"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises((TypeError, AttributeError)):
                DBLoader(db_uri="mongodb://localhost:27017")