FROM bitnami/spark:latest

# Switch to root to install Python, NumPy, and Streamlit
USER root

# Update package list, install Python3-pip, install packages, then clean up
RUN apt-get update && \
    apt-get install -y python3-pip && \
    pip3 install numpy streamlit && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Switch back to non-root user
USER 1001
