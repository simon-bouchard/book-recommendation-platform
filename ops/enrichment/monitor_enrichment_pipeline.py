#!/usr/bin/env python3
# ops/monitor_enrichment_pipeline.py
"""
Monitoring dashboard for Phase 1-2 enrichment pipeline.
Shows Kafka consumer lag, error rates, SQL commit times, and top issues.
NOW VERSION-AWARE: Track v1, v2, or all versions.
"""

import os
import time
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse

import pymysql
from dotenv import load_dotenv
from kafka import KafkaAdminClient, KafkaConsumer
from kafka.structs import TopicPartition

load_dotenv()

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
JDBC_URL = os.getenv("JDBC_URL", "jdbc:mysql://127.0.0.1:3306/bookrec_db")
JDBC_USER = os.getenv("JDBC_USER", "bookrec")
JDBC_PASS = os.getenv("JDBC_PASS", "secret")


class PipelineMonitor:
    def __init__(self, tags_version: Optional[str] = None):
        """
        Initialize pipeline monitor.

        Args:
            tags_version: Specific version to monitor (e.g., 'v2'), or None for all versions
        """
        self.tags_version = tags_version

        self.admin = KafkaAdminClient(
            bootstrap_servers=KAFKA_BOOTSTRAP, client_id="pipeline-monitor"
        )

        # Parse JDBC URL
        parsed = urlparse(JDBC_URL.replace("jdbc:", "", 1))
        self.db_conn = pymysql.connect(
            host=parsed.hostname or "127.0.0.1",
            port=parsed.port or 3306,
            user=JDBC_USER,
            password=JDBC_PASS,
            database=parsed.path.lstrip("/") or "bookrec_db",
            charset="utf8mb4",
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
        """Get error statistics from SQL, filtered by version if specified"""
        cur = self.db_conn.cursor(pymysql.cursors.DictCursor)

        # Build version filter
        version_filter = ""
        if self.tags_version:
            version_filter = f"AND tags_version = '{self.tags_version}'"

        # Error rate by stage/code
        cur.execute(f"""
            SELECT
                stage,
                error_code,
                tags_version,
                COUNT(*) as count,
                MAX(last_seen_at) as latest
            FROM enrichment_errors
            WHERE last_seen_at >= DATE_SUB(NOW(), INTERVAL {hours} HOUR)
            {version_filter}
            GROUP BY stage, error_code, tags_version
            ORDER BY count DESC
            LIMIT 10
        """)
        error_rates = cur.fetchall()

        # Summary
        cur.execute(f"""
            SELECT
                COUNT(DISTINCT item_idx) as total_failed,
                SUM(occurrence_count) as total_occurrences,
                MAX(last_seen_at) as latest_error
            FROM enrichment_errors
            WHERE 1=1 {version_filter}
        """)
        summary = cur.fetchone()

        # Top offenders
        cur.execute(f"""
            SELECT
                item_idx,
                title,
                occurrence_count,
                error_code,
                tags_version,
                last_seen_at
            FROM enrichment_errors
            WHERE 1=1 {version_filter}
            ORDER BY occurrence_count DESC
            LIMIT 5
        """)
        top_offenders = cur.fetchall()

        cur.close()
        return error_rates, summary, top_offenders

    def get_throughput_stats(self):
        """Get enrichment throughput stats, with version breakdown"""
        cur = self.db_conn.cursor(pymysql.cursors.DictCursor)

        # Count total books
        cur.execute("SELECT COUNT(*) as count FROM books WHERE item_idx IS NOT NULL")
        total = cur.fetchone()["count"]

        # Get breakdown by version
        cur.execute("""
            SELECT
                tags_version,
                COUNT(DISTINCT item_idx) as count
            FROM book_llm_subjects
            GROUP BY tags_version
            ORDER BY tags_version
        """)
        version_breakdown = cur.fetchall()

        # If specific version requested, get just that count
        if self.tags_version:
            cur.execute(f"""
                SELECT COUNT(DISTINCT item_idx) as count
                FROM book_llm_subjects
                WHERE tags_version = '{self.tags_version}'
            """)
            enriched = cur.fetchone()["count"]
        else:
            # All versions combined
            enriched = sum(row["count"] for row in version_breakdown)

        cur.close()

        coverage = (enriched / total * 100) if total > 0 else 0
        return enriched, total, coverage, version_breakdown

    def print_dashboard(self):
        """Print monitoring dashboard"""
        print("\n" + "=" * 80)
        version_tag = f" [{self.tags_version}]" if self.tags_version else " [ALL VERSIONS]"
        print(
            f"ENRICHMENT PIPELINE MONITORING{version_tag} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        print("=" * 80)

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
        enriched, total, coverage, version_breakdown = self.get_throughput_stats()

        if self.tags_version:
            print(
                f"  Enriched Books ({self.tags_version}): {enriched:,} / {total:,} ({coverage:.1f}%)"
            )
        else:
            print(f"  Enriched Books (All): {enriched:,} / {total:,} ({coverage:.1f}%)")

            if version_breakdown:
                print("\n  Breakdown by Version:")
                for row in version_breakdown:
                    version_coverage = (row["count"] / total * 100) if total > 0 else 0
                    print(
                        f"    {row['tags_version']:8} | {row['count']:>7,} books ({version_coverage:>5.1f}%)"
                    )

        # Errors
        print("\n⚠️  ERROR SUMMARY (Last 24h)")
        print("-" * 80)
        error_rates, summary, top_offenders = self.get_error_stats(24)

        if summary["total_failed"]:
            print(f"  Total Failed Books: {summary['total_failed']:,}")
            print(f"  Total Occurrences: {summary['total_occurrences']:,}")
            print(f"  Latest Error: {summary['latest_error']}")

            print("\n  Top Error Types:")
            for err in error_rates[:5]:
                version_tag = f"[{err['tags_version']}]" if not self.tags_version else ""
                print(
                    f"    {err['stage']:15} | {err['error_code']:20} {version_tag:6} | Count: {err['count']:>4}"
                )

            print("\n  Top Offenders (by occurrence):")
            for book in top_offenders:
                title = (book["title"] or "")[:40]
                version_tag = f"[{book['tags_version']}]" if not self.tags_version else ""
                print(
                    f"    item_idx={book['item_idx']:>6} | {title:40} {version_tag:6} | {book['occurrence_count']:>3}x | {book['error_code']}"
                )
        else:
            print("  ✓ No errors in last 24 hours!")

        print("\n" + "=" * 80 + "\n")

    def close(self):
        self.admin.close()
        self.db_conn.close()


def monitor_loop(interval_seconds: int = 30, tags_version: Optional[str] = None):
    """Run monitoring loop"""
    monitor = PipelineMonitor(tags_version=tags_version)

    try:
        while True:
            monitor.print_dashboard()
            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        print("\nStopping monitor...")
    finally:
        monitor.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Monitor enrichment pipeline with optional version filtering"
    )
    parser.add_argument(
        "--version", help="Monitor specific tags version (e.g., v2), or omit for all versions"
    )
    parser.add_argument("--loop", action="store_true", help="Run continuous monitoring loop")
    parser.add_argument(
        "--interval", type=int, default=30, help="Loop interval in seconds (default: 30)"
    )

    args = parser.parse_args()

    if args.loop:
        # Continuous monitoring
        monitor_loop(args.interval, args.version)
    else:
        # One-shot dashboard
        monitor = PipelineMonitor(tags_version=args.version)
        try:
            monitor.print_dashboard()
        finally:
            monitor.close()
