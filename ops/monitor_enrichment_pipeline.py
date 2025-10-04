#!/usr/bin/env python3
# ops/monitor_enrichment_pipeline.py
"""
Monitoring dashboard for Phase 1 enrichment pipeline.
Shows Kafka consumer lag, error rates, SQL commit times, and top issues.
"""
import os
import time
from datetime import datetime, timedelta
from kafka import KafkaConsumer, KafkaAdminClient
from kafka.structs import TopicPartition
import pymysql
from urllib.parse import urlparse

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
JDBC_URL = os.getenv("JDBC_URL", "jdbc:mysql://127.0.0.1:3306/bookrec_db")
JDBC_USER = os.getenv("JDBC_USER", "bookrec")
JDBC_PASS = os.getenv("JDBC_PASS", "secret")

class PipelineMonitor:
    def __init__(self):
        self.admin = KafkaAdminClient(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            client_id='pipeline-monitor'
        )
        
        # Parse JDBC URL
        parsed = urlparse(JDBC_URL.replace("jdbc:", "", 1))
        self.db_conn = pymysql.connect(
            host=parsed.hostname or "127.0.0.1",
            port=parsed.port or 3306,
            user=JDBC_USER,
            password=JDBC_PASS,
            database=parsed.path.lstrip("/") or "bookrec_db",
            charset="utf8mb4"
        )
    
    def get_consumer_lag(self, topic: str, group_id: str):
        """
        Calculate consumer lag for a topic/group.
        Returns total lag across all partitions.
        """
        consumer = KafkaConsumer(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            group_id=group_id,
            enable_auto_commit=False,
        )
        
        try:
            # Get partitions for topic
            partitions = consumer.partitions_for_topic(topic)
            if not partitions:
                return 0, {}
            
            topic_partitions = [TopicPartition(topic, p) for p in partitions]
            
            # Get committed offsets
            committed = consumer.committed_offsets(topic_partitions)
            
            # Get end offsets (latest)
            end_offsets = consumer.end_offsets(topic_partitions)
            
            # Calculate lag per partition
            lag_by_partition = {}
            total_lag = 0
            
            for tp in topic_partitions:
                committed_offset = committed.get(tp, 0)
                end_offset = end_offsets.get(tp, 0)
                lag = max(0, end_offset - committed_offset)
                lag_by_partition[tp.partition] = lag
                total_lag += lag
            
            return total_lag, lag_by_partition
            
        finally:
            consumer.close()
    
    def get_error_stats(self, hours: int = 24):
        """Get error statistics from SQL"""
        cur = self.db_conn.cursor(pymysql.cursors.DictCursor)
        
        # Error rate by stage/code
        cur.execute(f"""
            SELECT 
                stage,
                error_code,
                COUNT(*) as count,
                MAX(last_seen_at) as latest
            FROM enrichment_errors
            WHERE last_seen_at >= DATE_SUB(NOW(), INTERVAL {hours} HOUR)
            GROUP BY stage, error_code
            ORDER BY count DESC
            LIMIT 10
        """)
        error_rates = cur.fetchall()
        
        # Summary
        cur.execute("""
            SELECT 
                COUNT(DISTINCT item_idx) as total_failed,
                SUM(occurrence_count) as total_occurrences,
                MAX(last_seen_at) as latest_error
            FROM enrichment_errors
        """)
        summary = cur.fetchone()
        
        # Top offenders
        cur.execute("""
            SELECT 
                item_idx,
                title,
                occurrence_count,
                error_code,
                last_seen_at
            FROM enrichment_errors
            ORDER BY occurrence_count DESC
            LIMIT 5
        """)
        top_offenders = cur.fetchall()
        
        cur.close()
        return error_rates, summary, top_offenders
    
    def get_throughput_stats(self):
        """Get enrichment throughput stats"""
        cur = self.db_conn.cursor(pymysql.cursors.DictCursor)
        
        # Count enriched books
        cur.execute("SELECT COUNT(DISTINCT item_idx) as count FROM book_llm_subjects")
        enriched = cur.fetchone()['count']
        
        # Count total books
        cur.execute("SELECT COUNT(*) as count FROM books WHERE item_idx IS NOT NULL")
        total = cur.fetchone()['count']
        
        cur.close()
        
        coverage = (enriched / total * 100) if total > 0 else 0
        return enriched, total, coverage
    
    def print_dashboard(self):
        """Print monitoring dashboard"""
        print("\n" + "="*80)
        print(f"ENRICHMENT PIPELINE MONITORING - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        # Kafka Consumer Lag
        print("\n📊 KAFKA CONSUMER LAG")
        print("-" * 80)
        
        for topic, group in [
            ("enrich.results.v1", "cg.enrichment.sql.v1"),
            ("enrich.errors.v1", "cg.enrichment.errors.v1"),
        ]:
            try:
                total_lag, lag_by_partition = self.get_consumer_lag(topic, group)
                print(f"  {topic:30} | Lag: {total_lag:>6} messages")
                if total_lag > 0:
                    for p, lag in lag_by_partition.items():
                        if lag > 0:
                            print(f"    └─ Partition {p}: {lag} messages")
            except Exception as e:
                print(f"  {topic:30} | Error: {e}")
        
        # Throughput
        print("\n📈 THROUGHPUT")
        print("-" * 80)
        enriched, total, coverage = self.get_throughput_stats()
        print(f"  Enriched Books: {enriched:,} / {total:,} ({coverage:.1f}%)")
        
        # Errors
        print("\n⚠️  ERROR SUMMARY (Last 24h)")
        print("-" * 80)
        error_rates, summary, top_offenders = self.get_error_stats(24)
        
        if summary['total_failed']:
            print(f"  Total Failed Books: {summary['total_failed']:,}")
            print(f"  Total Occurrences: {summary['total_occurrences']:,}")
            print(f"  Latest Error: {summary['latest_error']}")
            
            print("\n  Top Error Types:")
            for err in error_rates[:5]:
                print(f"    {err['stage']:15} | {err['error_code']:20} | Count: {err['count']:>4}")
            
            print("\n  Top Offenders (by occurrence):")
            for book in top_offenders:
                title = (book['title'] or '')[:40]
                print(f"    item_idx={book['item_idx']:>6} | {title:40} | {book['occurrence_count']:>3}x | {book['error_code']}")
        else:
            print("  ✓ No errors in last 24 hours!")
        
        print("\n" + "="*80 + "\n")
    
    def close(self):
        self.admin.close()
        self.db_conn.close()


def monitor_loop(interval_seconds: int = 30):
    """Run monitoring loop"""
    monitor = PipelineMonitor()
    
    try:
        while True:
            monitor.print_dashboard()
            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        print("\nStopping monitor...")
    finally:
        monitor.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--loop":
        # Continuous monitoring
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        monitor_loop(interval)
    else:
        # One-shot dashboard
        monitor = PipelineMonitor()
        try:
            monitor.print_dashboard()
        finally:
            monitor.close() 
