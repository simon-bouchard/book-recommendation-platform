FROM python:3.10-slim

# Java for Spark + certs
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless curl ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Python deps for the Spark job
RUN pip install --no-cache-dir \
    pyspark==3.5.1 pyarrow pandas pymysql

# MySQL JDBC driver
ENV MYSQL_JDBC_VERSION=8.4.0
RUN mkdir -p /opt/jdbc && \
    curl -L -o /opt/jdbc/mysql-connector-j-${MYSQL_JDBC_VERSION}.jar \
    https://repo1.maven.org/maven2/com/mysql/mysql-connector-j/${MYSQL_JDBC_VERSION}/mysql-connector-j-${MYSQL_JDBC_VERSION}.jar

# Expose the JDBC jar to Spark
ENV SPARK_CLASSPATH=/opt/jdbc/mysql-connector-j-${MYSQL_JDBC_VERSION}.jar

# Java home for Spark
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

WORKDIR /app

# Default command (compose overrides too)
CMD ["bash", "-lc", "python /app/load_enrichment_to_sql.py"]

