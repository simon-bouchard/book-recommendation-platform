#!/usr/bin/env python3
# ops/setup_kafka_topics.py
"""
Create Kafka topics for enrichment pipeline with proper configuration.
Run this after starting Kafka: python ops/setup_kafka_topics.py
"""
import os
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.admin import ConfigResource, ConfigResourceType

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

def create_topics():
    admin = KafkaAdminClient(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        client_id='enrichment-topic-creator'
    )
    
    topics = [
        NewTopic(
            name="enrich.results.v1",
            num_partitions=6,
            replication_factor=1,
            topic_configs={
                # Compact old records but keep recent ones
                'cleanup.policy': 'compact,delete',
                # Keep records for 7 days
                'retention.ms': str(7 * 24 * 60 * 60 * 1000),
                # Wait 1 hour before compacting to allow batch processing
                'min.compaction.lag.ms': str(60 * 60 * 1000),
                # Compression for efficient storage
                'compression.type': 'snappy',
                # Segment size (100 MB)
                'segment.ms': str(60 * 60 * 1000),
                'segment.bytes': str(100 * 1024 * 1024),
            }
        ),
        NewTopic(
            name="enrich.errors.v1",
            num_partitions=3,
            replication_factor=1,
            topic_configs={
                # Delete-only for DLQ
                'cleanup.policy': 'delete',
                # Keep errors for 14 days
                'retention.ms': str(14 * 24 * 60 * 60 * 1000),
                'compression.type': 'snappy',
            }
        ),
    ]
    
    try:
        admin.create_topics(new_topics=topics, validate_only=False)
        print("✓ Topics created successfully:")
        print("  - enrich.results.v1 (6 partitions, 7d retention, compact+delete)")
        print("  - enrich.errors.v1 (3 partitions, 14d retention, delete)")
    except Exception as e:
        if "TopicExistsException" in str(e):
            print("ℹ Topics already exist")
        else:
            raise
    finally:
        admin.close()

def verify_topics():
    """Verify topics exist and show their configuration"""
    admin = KafkaAdminClient(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        client_id='enrichment-topic-verifier'
    )
    
    try:
        metadata = admin.list_topics()
        enrichment_topics = [t for t in metadata if t.startswith('enrich.')]
        
        if not enrichment_topics:
            print("⚠ No enrichment topics found!")
            return False
        
        print(f"\n✓ Found {len(enrichment_topics)} enrichment topics:")
        for topic in enrichment_topics:
            print(f"  - {topic}")
        
        return True
    finally:
        admin.close()

if __name__ == "__main__":
    print("Setting up Kafka topics for enrichment pipeline...\n")
    create_topics()
    print()
    verify_topics()
