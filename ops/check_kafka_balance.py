#!/usr/bin/env python3
# ops/check_kafka_balance.py
"""
Check Kafka partition balance and consumer health.
Shows message distribution, lag, and imbalance metrics.
"""
import sys
import os
from kafka import KafkaConsumer, KafkaAdminClient
from kafka.structs import TopicPartition
from collections import defaultdict
import statistics

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")


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
        ideal_per_partition = total / partition_count if partition_count > 0 else 0
        
        values = list(counts.values())
        mean = statistics.mean(values) if values else 0
        stdev = statistics.stdev(values) if len(values) > 1 else 0
        min_val = min(values) if values else 0
        max_val = max(values) if values else 0
        
        # Coefficient of variation (normalized std dev)
        cv = (stdev / mean * 100) if mean > 0 else 0
        
        # Max deviation from ideal
        max_deviation = max(abs(count - ideal_per_partition) for count in values) if values else 0
        max_deviation_pct = (max_deviation / ideal_per_partition * 100) if ideal_per_partition > 0 else 0
        
        # Print results
        print(f"\n{'='*70}")
        print(f"Topic: {topic}")
        print(f"{'='*70}")
        print(f"Partitions: {partition_count}")
        print(f"Total messages: {total:,}")
        print(f"Ideal per partition: {ideal_per_partition:,.1f}")
        print(f"\nDistribution statistics:")
        print(f"  Mean: {mean:,.1f}")
        print(f"  Std Dev: {stdev:,.1f}")
        print(f"  Coefficient of Variation: {cv:.1f}%")
        print(f"  Min: {min_val:,} | Max: {max_val:,}")
        print(f"  Range: {max_val - min_val:,} ({(max_val - min_val) / total * 100:.1f}% of total)")
        
        print(f"\nPartition breakdown:")
        for p, count in sorted(counts.items()):
            pct = (count / total * 100) if total > 0 else 0
            deviation = count - ideal_per_partition
            deviation_pct = (deviation / ideal_per_partition * 100) if ideal_per_partition > 0 else 0
            
            # Visual bar (scaled to 50 chars max)
            bar_length = int((count / max_val * 50)) if max_val > 0 else 0
            bar = '█' * bar_length
            
            status = ""
            if abs(deviation_pct) > 50:  # More than 50% deviation
                status = " ⚠️  SEVERELY IMBALANCED"
            elif abs(deviation_pct) > 25:  # More than 25% deviation
                status = " ⚠️  IMBALANCED"
            elif verbose and abs(deviation_pct) < 10:
                status = " ✓"
            
            print(f"  Partition {p:2d}: {count:>10,} ({pct:>5.1f}%) "
                  f"[{deviation:>+8,.0f}] {bar}{status}")
        
        # Assessment
        print(f"\n{'='*70}")
        if cv < 10:
            print("✅ EXCELLENT: Partitions are well-balanced (CV < 10%)")
        elif cv < 20:
            print("✓ GOOD: Acceptable balance (CV < 20%)")
        elif cv < 30:
            print("⚠️  WARNING: Noticeable imbalance (CV < 30%)")
        else:
            print("❌ POOR: Significant imbalance (CV >= 30%)")
        
        print(f"\nBalance assessment:")
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
            print("    4. Check for skewed data (are certain item_idx ranges over-represented?)")
        
        print(f"{'='*70}\n")
        
        return {
            "topic": topic,
            "partition_count": partition_count,
            "total_messages": total,
            "cv": cv,
            "max_deviation_pct": max_deviation_pct,
            "balanced": cv < 20,
        }
        
    except Exception as e:
        print(f"❌ Error checking topic '{topic}': {e}")
        return None


def check_consumer_lag(topic: str, group_id: str):
    """
    Check consumer lag for a consumer group.
    Shows how far behind consumers are.
    """
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
        
        # Get committed offsets (where consumer is)
        committed = consumer.committed_offsets(topic_partitions)
        
        # Get end offsets (latest in topic)
        end_offsets = consumer.end_offsets(topic_partitions)
        
        print(f"\nConsumer Lag: {group_id}")
        print(f"{'-'*70}")
        
        total_lag = 0
        for tp in topic_partitions:
            committed_offset = committed.get(tp, 0) or 0
            end_offset = end_offsets.get(tp, 0)
            lag = max(0, end_offset - committed_offset)
            total_lag += lag
            
            if lag > 0:
                print(f"  Partition {tp.partition}: Lag = {lag:,} messages")
        
        if total_lag == 0:
            print(f"  ✅ No lag - consumer is caught up")
        elif total_lag < 1000:
            print(f"\n  ✓ Total lag: {total_lag:,} messages (healthy)")
        elif total_lag < 10000:
            print(f"\n  ⚠️  Total lag: {total_lag:,} messages (monitor)")
        else:
            print(f"\n  ❌ Total lag: {total_lag:,} messages (action needed)")
            print(f"     Consider increasing consumer parallelism")
        
        consumer.close()
        return total_lag
        
    except Exception as e:
        print(f"❌ Error checking consumer lag: {e}")
        return None


def main():
    """Run all checks"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Check Kafka partition balance and consumer health")
    parser.add_argument(
        "--topics",
        nargs="+",
        default=["enrich.results.v1", "enrich.errors.v1"],
        help="Topics to check"
    )
    parser.add_argument(
        "--check-lag",
        action="store_true",
        help="Also check consumer lag"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show more details"
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("KAFKA PARTITION BALANCE CHECK")
    print("="*70)
    
    results = []
    for topic in args.topics:
        result = check_partition_balance(topic, verbose=args.verbose)
        if result:
            results.append(result)
    
    # Consumer lag checks
    if args.check_lag:
        print("\n" + "="*70)
        print("CONSUMER LAG CHECK")
        print("="*70)
        check_consumer_lag("enrich.results.v1", "cg.enrichment.sql.v1")
        check_consumer_lag("enrich.errors.v1", "cg.enrichment.errors.v1")
    
    # Summary
    if results:
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        
        all_balanced = all(r["balanced"] for r in results)
        if all_balanced:
            print("✅ All topics are well-balanced")
            return 0
        else:
            print("⚠️  Some topics have imbalance - review details above")
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
