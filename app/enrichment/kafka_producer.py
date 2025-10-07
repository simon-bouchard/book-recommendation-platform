# app/enrichment/kafka_producer.py
"""
Kafka producer for enrichment pipeline with retry logic and error handling.
Supports dual-write mode (Kafka + JSONL) during cutover.
"""
import os
import json
import time
from typing import Dict, Any, Optional
from pathlib import Path
from kafka import KafkaProducer
from kafka.errors import KafkaError, KafkaTimeoutError
import logging

logger = logging.getLogger(__name__)

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
RESULTS_TOPIC = "enrich.results.v1"
ERRORS_TOPIC = "enrich.errors.v1"

# Feature flag: dual-write to both Kafka and JSONL during cutover
DUAL_WRITE_JSONL = os.getenv("ENRICH_DUAL_WRITE_JSONL", "1") == "1"
JSONL_PATH = Path(os.getenv("ENRICH_JSONL_PATH", "data/enrichment_v1.jsonl"))


class EnrichmentProducer:
    """
    Kafka producer for enrichment events with retry logic and fallback to local queue.
    """
    
    def __init__(self, enable_kafka: bool = True):
        self.enable_kafka = enable_kafka and KAFKA_BOOTSTRAP
        self.producer: Optional[KafkaProducer] = None
        self.circuit_open = False
        self.failure_count = 0
        self.last_failure_time = 0
        
        # Circuit breaker config
        self.max_failures = 3
        self.circuit_timeout = 60  # seconds
        
        if self.enable_kafka:
            self._init_producer()
    
    def _init_producer(self):
        """Initialize Kafka producer with retry logic"""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP,
                value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode('utf-8'),
                key_serializer=lambda k: str(k).encode('utf-8') if k else None,
                # Retry settings
                retries=3,
                retry_backoff_ms=100,
                max_in_flight_requests_per_connection=5,
                # Batching for efficiency
                linger_ms=100,
                batch_size=32768,
                # Compression
                compression_type='snappy',
                # Timeouts
                request_timeout_ms=30000,
                # Acknowledgment
                acks='all',
            )
            logger.info("Kafka producer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Kafka producer: {e}")
            self.enable_kafka = False
            self.circuit_open = True
    
    def _check_circuit(self) -> bool:
        """Check if circuit breaker should allow attempts"""
        if not self.circuit_open:
            return True
        
        # Check if timeout has passed
        if time.time() - self.last_failure_time > self.circuit_timeout:
            logger.info("Circuit breaker timeout passed, attempting to reconnect")
            self.circuit_open = False
            self.failure_count = 0
            self._init_producer()
            return True
        
        return False
    
    def _record_failure(self):
        """Record a failure and potentially open circuit breaker"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.max_failures:
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
            self.circuit_open = True
    
    def _record_success(self):
        """Record a success and reset failure count"""
        if self.failure_count > 0:
            logger.info("Kafka connection recovered")
        self.failure_count = 0
    
    def send_result(
        self,
        item_idx: int,
        subjects: list[str],
        tone_ids: list[int],
        genre: str,
        vibe: str,
        tags_version: str = "v1",
        scores: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Send successful enrichment result to Kafka.
        
        Returns:
            bool: True if sent successfully (or Kafka disabled), False if failed
        """
        event = {
            "item_idx": item_idx,
            "subjects": subjects,
            "tone_ids": tone_ids,
            "genre": genre,
            "vibe": vibe,
            "tags_version": tags_version,
            "scores": scores or {},
            "timestamp": int(time.time() * 1000),
        }
        
        if metadata:
            event["metadata"] = metadata
        
        if not self.enable_kafka:
            raise RuntimeError("Kafka disabled - enrichment cannot proceed")
        
        if not self._check_circuit():
            raise RuntimeError(
                f"Circuit breaker open after {self.failure_count} failures. "
                f"Last failure {time.time() - self.last_failure_time:.0f}s ago. "
                "Fix Kafka before continuing."
            )
        
        # Single source of truth
        success = self._send_to_kafka(...)
        if not success:
            raise RuntimeError(f"Kafka send failed for item_idx={item_idx}")
        
        return True

    def send_error(
        self,
        item_idx: int,
        error_msg: str,
        stage: str,
        error_code: str,
        tags_version: str = "v1",
        error_field: Optional[str] = None,
        title: Optional[str] = None,
        author: Optional[str] = None,
        attempted: Optional[Dict[str, Any]] = None,
        run_metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Send error event to DLQ topic.
        
        Args:
            item_idx: Book item_idx
            error_msg: Error message (truncated to 512 chars)
            stage: Pipeline stage (fetch|llm_invoke|llm_parse|validate|postprocess|produce)
            error_code: Error code (INVALID_GENRE, TIMEOUT, etc.)
            error_field: Specific field that failed validation
            title: Book title (truncated to 256 chars)
            author: Author name (truncated to 256 chars)
            attempted: Partial results that were attempted
            run_metadata: Run info (run_id, model, provider, latency_ms)
        """
        event = {
            "item_idx": item_idx,
            "tags_version": tags_version,
            "timestamp": int(time.time() * 1000),
            "stage": stage,
            "error_code": error_code,
            "error_msg": (error_msg or "")[:512],
        }
        
        if error_field:
            event["error_field"] = error_field
        if title:
            event["title"] = title[:256]
        if author:
            event["author"] = author[:256]
        if attempted:
            event["attempted"] = attempted
        if run_metadata:
            event.update(run_metadata)
        
        # Dual-write to JSONL if enabled
        if DUAL_WRITE_JSONL:
            jsonl_event = {"book_id": item_idx, "error": error_msg, "tags_version": tags_version}
            self._write_jsonl(jsonl_event)
        
        # Send to Kafka if enabled
        if self.enable_kafka and self._check_circuit():
            return self._send_to_kafka(ERRORS_TOPIC, item_idx, tags_version, event)
        
        return True
    
    def _send_to_kafka(self, topic: str, item_idx: int, tags_version: str, event: dict) -> bool:
        """Send event to Kafka with retry logic"""
        key = f"{item_idx}:{tags_version}"
        
        try:
            future = self.producer.send(topic, key=key, value=event)
            # Wait for acknowledgment (with timeout)
            future.get(timeout=10)
            self._record_success()
            return True
            
        except KafkaTimeoutError as e:
            logger.warning(f"Kafka timeout for {key}: {e}")
            self._record_failure()
            return False
            
        except KafkaError as e:
            logger.error(f"Kafka error for {key}: {e}")
            self._record_failure()
            return False
            
        except Exception as e:
            logger.error(f"Unexpected error sending to Kafka for {key}: {e}")
            self._record_failure()
            return False
    
    def _write_jsonl(self, event: dict):
        """Fallback/dual-write to JSONL file"""
        try:
            JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(JSONL_PATH, "a", encoding="utf-8") as f:
                # Convert to legacy format for compatibility
                legacy_event = {
                    "book_id": event.get("item_idx"),
                    "subjects": event.get("subjects", []),
                    "tone_ids": event.get("tone_ids", []),
                    "genre": event.get("genre", ""),
                    "vibe": event.get("vibe", ""),
                    "tags_version": event.get("tags_version", "v1"),
                    "scores": event.get("scores", {}),
                }
                if "error" in event:
                    legacy_event = {"book_id": event["item_idx"], "error": event["error"], "tags_version": event["tags_version"]}
                
                f.write(json.dumps(legacy_event, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Failed to write JSONL: {e}")
    
    def flush(self):
        """Flush any pending messages"""
        if self.producer:
            try:
                self.producer.flush(timeout=10)
            except Exception as e:
                logger.error(f"Error flushing producer: {e}")
    
    def close(self):
        """Close producer and release resources"""
        if self.producer:
            try:
                self.producer.flush(timeout=10)
                self.producer.close(timeout=10)
                logger.info("Kafka producer closed")
            except Exception as e:
                logger.error(f"Error closing producer: {e}")
