# Use official lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install OS dependencies
RUN set -ex \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    git \
    libgl1-mesa-glx \
    wget \
    libglib2.0-0 \
    openjdk-17-jdk \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the app files
COPY . .
# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose the Streamlit default port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
