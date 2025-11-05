#!/bin/bash
set -e

echo "🧹 Clearing Kafka Test Data"
echo "========================================"

# Stop consumers
echo "1. Stopping Spark streaming consumers..."
docker-compose -f docker/spark-loader/docker-compose.yml down 2>/dev/null || true

# Delete and recreate topics
echo ""
echo "2. Deleting Kafka topics..."
docker exec kafka kafka-topics --bootstrap-server localhost:9092 \
  --delete --topic enrich.results.v1 2>/dev/null || true

docker exec kafka kafka-topics --bootstrap-server localhost:9092 \
  --delete --topic enrich.errors.v1 2>/dev/null || true

echo "   Waiting for deletion to complete..."
sleep 5

# Recreate
echo ""
echo "3. Recreating topics..."
python ops/setup_kafka_topics.py

# Delete Spark checkpoints (so it doesn't resume from old offsets)
echo ""
echo "4. Clearing Spark checkpoints..."
rm -rf /tmp/spark-checkpoints/enrichment/v2/ 2>/dev/null || true

# Verify
echo ""
echo "5. Verifying topics..."
docker exec kafka kafka-topics --bootstrap-server localhost:9092 --list | grep enrich

echo ""
echo "✅ Kafka test data cleared!"
echo ""
echo "Next steps:"
echo "  1. Restart Spark consumer: docker-compose -f docker/spark-loader/docker-compose.yml up"
echo "  2. Re-run enrichment: python -m app.enrichment.runner_kafka --limit 10"
echo "========================================"
