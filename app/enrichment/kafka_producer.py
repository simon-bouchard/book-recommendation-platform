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
CUTOVER_COMPLETE = os.getenv("ENRICH_CUTOVER_COMPLETE", "0") == "1"
JSONL_PATH = Path(os.getenv("ENRICH_JSONL_PATH", "data/enrichment_v1.jsonl"))


class EnrichmentProducer:
    """
    Kafka producer for enrichment pipeline.
    
    Modes:
      - Cutover (DUAL_WRITE_JSONL=1, CUTOVER_COMPLETE=0): 
        Write to both Kafka and JSONL for validation
      - Post-cutover (CUTOVER_COMPLETE=1):
        Kafka only, fail fast if unavailable
    """
    
    def __init__(self, enable_kafka: bool = True):
        self.enable_kafka = enable_kafka and KAFKA_BOOTSTRAP
        self.producer: Optional[KafkaProducer] = None
        self.circuit_open = False
        self.failure_count = 0
        self.last_failure_time = 0
        
        # Circuit breaker config
        self.max_failures = 3
        self.circuit_timeout = 60
        
        # Validate configuration
        if CUTOVER_COMPLETE and not self.enable_kafka:
            raise RuntimeError(
                "ENRICH_CUTOVER_COMPLETE=1 but Kafka is disabled. "
                "Cannot proceed without Kafka after cutover."
            )
        
        if self.enable_kafka:
            self._init_producer()
        elif not DUAL_WRITE_JSONL:
            raise RuntimeError(
                "Kafka disabled and DUAL_WRITE_JSONL=0. "
                "No data path available!"
            )
    
    def _init_producer(self):
        """Initialize Kafka producer with retry logic"""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP,
                value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode('utf-8'),
                key_serializer=lambda k: str(k).encode('utf-8') if k else None,
                retries=3,
                retry_backoff_ms=100,
                max_in_flight_requests_per_connection=5,
                linger_ms=100,
                batch_size=32768,
                compression_type='snappy',
                request_timeout_ms=30000,
                acks='all',
            )
            logger.info("Kafka producer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Kafka producer: {e}")
            if CUTOVER_COMPLETE:
                raise RuntimeError(
                    f"Cannot initialize Kafka producer after cutover: {e}"
                )
            self.enable_kafka = False
            self.circuit_open = True
    
    def _check_circuit(self) -> bool:
        """Check if circuit breaker allows attempts"""
        if not self.circuit_open:
            return True
        
        if time.time() - self.last_failure_time > self.circuit_timeout:
            logger.info("Circuit breaker timeout passed, attempting reconnect")
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
        Send enrichment result.
        
        Returns:
            bool: True if sent successfully
        
        Raises:
            RuntimeError: If send fails post-cutover
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
        
        # CUTOVER MODE: Dual-write for validation
        if DUAL_WRITE_JSONL and not CUTOVER_COMPLETE:
            self._write_jsonl(event)
        
        # Try Kafka
        if self.enable_kafka:
            if not self._check_circuit():
                msg = (
                    f"Circuit breaker open: {self.failure_count} failures, "
                    f"last {time.time() - self.last_failure_time:.0f}s ago"
                )
                if CUTOVER_COMPLETE:
                    raise RuntimeError(f"Kafka unavailable post-cutover: {msg}")
                else:
                    logger.warning(f"{msg} - relying on JSONL during cutover")
                    return True
            
            # ✅ FIX: Pass all required arguments
            kafka_ok = self._send_to_kafka(RESULTS_TOPIC, item_idx, tags_version, event)
            
            if not kafka_ok:
                if CUTOVER_COMPLETE:
                    raise RuntimeError(
                        f"Kafka send failed for item_idx={item_idx}. "
                        "Cannot proceed after cutover."
                    )
                else:
                    logger.warning(
                        f"Kafka send failed for item_idx={item_idx} "
                        "but JSONL has backup during cutover"
                    )
            
            return kafka_ok
        
        # Kafka disabled
        if CUTOVER_COMPLETE:
            raise RuntimeError("Kafka disabled but cutover complete - invalid state")
        
        logger.warning("Kafka disabled, using JSONL during cutover")
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
        """Send error event to DLQ"""
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
        
        # Dual-write during cutover
        if DUAL_WRITE_JSONL and not CUTOVER_COMPLETE:
            jsonl_event = {
                "book_id": item_idx,
                "error": error_msg,
                "tags_version": tags_version
            }
            self._write_jsonl(jsonl_event)
        
        # Send to Kafka
        if self.enable_kafka:
            if not self._check_circuit():
                if CUTOVER_COMPLETE:
                    raise RuntimeError("Circuit breaker open for errors post-cutover")
                return True
            
            # ✅ FIX: Pass all required arguments
            kafka_ok = self._send_to_kafka(ERRORS_TOPIC, item_idx, tags_version, event)
            if not kafka_ok and CUTOVER_COMPLETE:
                raise RuntimeError(f"Error DLQ send failed for item_idx={item_idx}")
            return kafka_ok
        
        if CUTOVER_COMPLETE:
            raise RuntimeError("Kafka disabled for errors post-cutover")
        
        return True
    
    def _send_to_kafka(self, topic: str, item_idx: int, tags_version: str, event: dict) -> bool:
        """Send event to Kafka with retry logic"""
        key = f"{item_idx}:{tags_version}"
        
        try:
            future = self.producer.send(topic, key=key, value=event)
            future.get(timeout=10)
            self._record_success()
            return True
            
        except (KafkaTimeoutError, KafkaError) as e:
            logger.error(f"Kafka error for {key}: {e}")
            self._record_failure()
            return False
            
        except Exception as e:
            logger.error(f"Unexpected error sending to Kafka for {key}: {e}")
            self._record_failure()
            return False
    
    def _write_jsonl(self, event: dict):
        """CUTOVER ONLY: Write to JSONL for validation"""
        try:
            JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(JSONL_PATH, "a", encoding="utf-8") as f:
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
                    legacy_event = {
                        "book_id": event["item_idx"],
                        "error": event["error"],
                        "tags_version": event["tags_version"]
                    }
                
                f.write(json.dumps(legacy_event, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Failed to write JSONL (cutover only): {e}")
    
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
