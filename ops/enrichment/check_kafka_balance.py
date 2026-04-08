#!/usr/bin/env python3
# ops/check_kafka_balance.py
"""
Check Kafka partition balance and consumer health.
Shows message distribution, lag, and imbalance metrics.
"""

import os
import statistics
import sys

from kafka import KafkaConsumer
from kafka.structs import TopicPartition

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
MINIMUM_SAMPLE_SIZE = 1000  # Need this many messages before balance matters


def check_partition_balance(topic: str, verbose: bool = False):
    """
    Analyze partition balance for a topic.
    Returns metrics dict for programmatic use.
    """
    try:
        consumer = KafkaConsumer(bootstrap_servers=KAFKA_BOOTSTRAP)
        partitions = consumer.partitions_for_topic(topic)

        if not partitions:
            print(f"❌ Topic '{topic}' not found or has no partitions")
            consumer.close()
            return None

        # Collect partition stats
        counts = {}
        for p in partitions:
            tp = TopicPartition(topic, p)
            start = consumer.beginning_offsets([tp])[tp]
            end = consumer.end_offsets([tp])[tp]
            counts[p] = end - start

        consumer.close()

        # Calculate statistics
        total = sum(counts.values())
        partition_count = len(partitions)

        # ✅ FIX 1: Handle empty topic
        if total == 0:
            print(f"\n{'=' * 70}")
            print(f"Topic: {topic}")
            print(f"{'=' * 70}")
            print(f"Partitions: {partition_count}")
            print("Total messages: 0")
            print("\n✓ Topic is empty - no balance to check")
            print(f"{'=' * 70}\n")
            return {
                "topic": topic,
                "partition_count": partition_count,
                "total_messages": 0,
                "cv": 0,
                "max_deviation_pct": 0,
                "balanced": True,
                "empty": True,
            }

        ideal_per_partition = total / partition_count

        values = list(counts.values())
        mean = statistics.mean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0
        min_val = min(values)
        max_val = max(values)

        # Coefficient of variation (normalized std dev)
        cv = (stdev / mean * 100) if mean > 0 else 0

        # Max deviation from ideal
        max_deviation = max(abs(count - ideal_per_partition) for count in values)
        max_deviation_pct = (
            (max_deviation / ideal_per_partition * 100) if ideal_per_partition > 0 else 0
        )

        # Print results
        print(f"\n{'=' * 70}")
        print(f"Topic: {topic}")
        print(f"{'=' * 70}")
        print(f"Partitions: {partition_count}")
        print(f"Total messages: {total:,}")
        print(f"Ideal per partition: {ideal_per_partition:,.1f}")

        # ✅ FIX 2: Warn about small sample size
        if total < MINIMUM_SAMPLE_SIZE:
            print(f"\n⚠️  WARNING: Sample size too small ({total} < {MINIMUM_SAMPLE_SIZE})")
            print("   Balance metrics are unreliable with few messages.")
            print(f"   Re-run after enriching at least {MINIMUM_SAMPLE_SIZE:,} books.\n")

        print("\nDistribution statistics:")
        print(f"  Mean: {mean:,.1f}")
        print(f"  Std Dev: {stdev:,.1f}")
        print(f"  Coefficient of Variation: {cv:.1f}%")
        print(f"  Min: {min_val:,} | Max: {max_val:,}")
        print(f"  Range: {max_val - min_val:,} ({(max_val - min_val) / total * 100:.1f}% of total)")

        print("\nPartition breakdown:")
        for p, count in sorted(counts.items()):
            pct = (count / total * 100) if total > 0 else 0
            deviation = count - ideal_per_partition
            deviation_pct = (
                (deviation / ideal_per_partition * 100) if ideal_per_partition > 0 else 0
            )

            # Visual bar (scaled to 50 chars max)
            bar_length = int((count / max_val * 50)) if max_val > 0 else 0
            bar = "█" * bar_length

            status = ""
            if total < MINIMUM_SAMPLE_SIZE:
                status = ""  # Don't flag when sample is too small
            elif abs(deviation_pct) > 50:
                status = " ⚠️  SEVERELY IMBALANCED"
            elif abs(deviation_pct) > 25:
                status = " ⚠️  IMBALANCED"
            elif verbose and abs(deviation_pct) < 10:
                status = " ✓"

            print(
                f"  Partition {p:2d}: {count:>10,} ({pct:>5.1f}%) "
                f"[{deviation:>+8,.0f}] {bar}{status}"
            )

        # Assessment
        print(f"\n{'=' * 70}")

        if total < MINIMUM_SAMPLE_SIZE:
            print(
                f"⏳ TOO EARLY: Need {MINIMUM_SAMPLE_SIZE - total:,} more messages for reliable assessment"
            )
        elif cv < 10:
            print("✅ EXCELLENT: Partitions are well-balanced (CV < 10%)")
        elif cv < 20:
            print("✓ GOOD: Acceptable balance (CV < 20%)")
        elif cv < 30:
            print("⚠️  WARNING: Noticeable imbalance (CV < 30%)")
        else:
            print("❌ POOR: Significant imbalance (CV >= 30%)")

        if total >= MINIMUM_SAMPLE_SIZE:
            print("\nBalance assessment:")
            if max_deviation_pct < 20:
                print("  No action needed - partitions are balanced")
            elif max_deviation_pct < 50:
                print("  Monitor - slight imbalance but acceptable")
                print("  Consider reviewing your partition key strategy")
            else:
                print("  Action required - significant imbalance detected")
                print("  Recommendations:")
                print("    1. Review partition key (currently: f'{item_idx}:{tags_version}')")
                print("    2. Consider: Use only item_idx as key (numeric distribution)")
                print("    3. Or: Increase partition count (currently: {})".format(partition_count))
                print(
                    "    4. Check for skewed data (are certain item_idx ranges over-represented?)"
                )

        print(f"{'=' * 70}\n")

        return {
            "topic": topic,
            "partition_count": partition_count,
            "total_messages": total,
            "cv": cv,
            "max_deviation_pct": max_deviation_pct,
            "balanced": cv < 20 or total < MINIMUM_SAMPLE_SIZE,  # Consider balanced if too early
            "sample_size_ok": total >= MINIMUM_SAMPLE_SIZE,
        }

    except Exception as e:
        print(f"❌ Error checking topic '{topic}': {e}")
        import traceback

        traceback.print_exc()
        return None


def check_spark_checkpoint_offsets(checkpoint_dir: str, topic: str):
    """
    Check Spark Structured Streaming checkpoint offsets.
    Spark stores offsets in checkpoint dir, not Kafka consumer groups.
    """
    import json
    from pathlib import Path

    offsets_dir = Path(checkpoint_dir) / "offsets"

    if not offsets_dir.exists():
        print(f"  ⚠️  Checkpoint directory not found: {checkpoint_dir}")
        print("     Spark may not have started yet")
        return None

    # Find latest offset file
    offset_files = sorted(offsets_dir.glob("*"))
    if not offset_files:
        print("  ⚠️  No offset files in checkpoint")
        return None

    latest_offset_file = offset_files[-1]

    try:
        with open(latest_offset_file, "r") as f:
            data = json.load(f)

        # Extract offsets for the topic
        topic_offsets = {}

        # Spark checkpoint format varies by version
        # Try common formats
        if topic in data:
            # Format 1: Direct topic key
            topic_offsets = data[topic]
        elif "offsets" in data and topic in data["offsets"]:
            # Format 2: Nested under "offsets"
            topic_offsets = data["offsets"][topic]
        else:
            # Search all keys
            for key, value in data.items():
                if isinstance(value, dict) and any(str(p).isdigit() for p in value.keys()):
                    topic_offsets = value
                    break

        if not topic_offsets:
            print(f"  ⚠️  No offsets found for topic '{topic}' in checkpoint")
            return None

        print(f"\nSpark Checkpoint Offsets: {checkpoint_dir}")
        print(f"{'-' * 70}")
        print(f"Latest checkpoint file: {latest_offset_file.name}")

        total_processed = 0
        for partition, offset in sorted(topic_offsets.items()):
            partition_num = int(partition) if isinstance(partition, str) else partition
            offset_num = int(offset)
            total_processed += offset_num
            print(f"  Partition {partition_num}: Offset {offset_num}")

        print(f"\n  ✓ Total messages processed: {total_processed:,}")

        return total_processed

    except Exception as e:
        print(f"  ❌ Error reading checkpoint: {e}")
        return None


def check_consumer_lag(topic: str, group_id: str, checkpoint_dir: str = None):
    """
    Check consumer lag for a consumer group OR Spark checkpoint.
    """

    # If this is a Spark consumer, check checkpoint instead
    if checkpoint_dir:
        return check_spark_checkpoint_offsets(checkpoint_dir, topic)

    try:
        consumer = KafkaConsumer(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            group_id=group_id,
            enable_auto_commit=False,
        )

        partitions = consumer.partitions_for_topic(topic)
        if not partitions:
            consumer.close()
            return None

        topic_partitions = [TopicPartition(topic, p) for p in partitions]

        # ✅ FIX 3: Use correct API - committed() not committed_offsets()
        # Get end offsets (latest in topic)
        end_offsets = consumer.end_offsets(topic_partitions)

        print(f"\nConsumer Lag: {group_id}")
        print(f"{'-' * 70}")

        total_lag = 0
        has_any_committed = False

        for tp in topic_partitions:
            # committed() returns OffsetAndMetadata or None
            committed_info = consumer.committed(tp)
            committed_offset = committed_info.offset if committed_info else None

            end_offset = end_offsets.get(tp, 0)

            if committed_offset is None:
                # Consumer group hasn't committed anything for this partition
                if end_offset > 0:
                    print(
                        f"  Partition {tp.partition}: Not started (no committed offset, {end_offset:,} messages waiting)"
                    )
                    total_lag += end_offset
            else:
                has_any_committed = True
                lag = max(0, end_offset - committed_offset)
                total_lag += lag

                if lag > 0:
                    print(f"  Partition {tp.partition}: Lag = {lag:,} messages")

        if not has_any_committed:
            print(f"  ⚠️  Consumer group '{group_id}' has never committed")
            print("     This is normal if the consumer hasn't started yet")
        elif total_lag == 0:
            print("  ✅ No lag - consumer is caught up")
        elif total_lag < 1000:
            print(f"\n  ✓ Total lag: {total_lag:,} messages (healthy)")
        elif total_lag < 10000:
            print(f"\n  ⚠️  Total lag: {total_lag:,} messages (monitor)")
        else:
            print(f"\n  ❌ Total lag: {total_lag:,} messages (action needed)")
            print("     Consider increasing consumer parallelism")

        consumer.close()
        return total_lag

    except Exception as e:
        print(f"❌ Error checking consumer lag: {e}")
        import traceback

        if "--verbose" in sys.argv:
            traceback.print_exc()
        return None


def main():
    """Run all checks"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Check Kafka partition balance and consumer health"
    )
    parser.add_argument(
        "--topics",
        nargs="+",
        default=["enrich.results.v1", "enrich.errors.v1"],
        help="Topics to check",
    )
    parser.add_argument("--check-lag", action="store_true", help="Also check consumer lag")
    parser.add_argument("--verbose", action="store_true", help="Show more details")

    args = parser.parse_args()

    print("=" * 70)
    print("KAFKA PARTITION BALANCE CHECK")
    print("=" * 70)

    results = []
    for topic in args.topics:
        result = check_partition_balance(topic, verbose=args.verbose)
        if result:
            results.append(result)

    # Consumer lag checks
    if args.check_lag:
        print("\n" + "=" * 70)
        print("CONSUMER LAG CHECK")
        print("=" * 70)

        # For Spark consumers, check checkpoint instead of Kafka consumer groups
        print("\n[Spark Consumer - Results]")
        check_consumer_lag(
            "enrich.results.v1",
            "cg.enrichment.sql.v1",
            checkpoint_dir="/tmp/spark-checkpoints/enrichment/v1/results",
        )

        print("\n[Spark Consumer - Errors]")
        check_consumer_lag(
            "enrich.errors.v1",
            "cg.enrichment.errors.v1",
            checkpoint_dir="/tmp/spark-checkpoints/enrichment/v1/errors",
        )

    # Summary
    if results:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        # Check if any have enough data
        any_with_data = any(r.get("sample_size_ok", False) for r in results)

        if not any_with_data:
            print("⏳ All topics have too few messages for reliable balance assessment")
            print(f"   Re-run after enriching at least {MINIMUM_SAMPLE_SIZE:,} books")
            return 0

        all_balanced = all(r["balanced"] for r in results if r.get("sample_size_ok", False))
        if all_balanced:
            print("✅ All topics are well-balanced")
            return 0
        else:
            print("⚠️  Some topics have imbalance - review details above")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
