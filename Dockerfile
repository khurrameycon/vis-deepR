FROM python:3.11-slim-bookworm

# Set build arguments
ARG TARGETPLATFORM=linux/amd64
ARG BUILD_DATE
ARG VERSION
ARG VCS_REF

# Add metadata
LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="browser-use-multi-user" \
      org.label-schema.description="Multi-user Browser Use API with session management" \
      org.label-schema.version=$VERSION \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/browser-use/web-ui" \
      org.label-schema.schema-version="1.0"

# Install system dependencies with proper cleanup
RUN apt-get update && apt-get install -y \
    wget \
    netcat-traditional \
    gnupg \
    curl \
    unzip \
    xvfb \
    libgconf-2-4 \
    libxss1 \
    libnss3 \
    libnspr4 \
    libasound2 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdbus-1-3 \
    libdrm2 \
    libgbm1 \
    libgtk-3-0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    xdg-utils \
    fonts-liberation \
    dbus \
    xauth \
    x11vnc \
    tigervnc-tools \
    supervisor \
    net-tools \
    procps \
    git \
    python3-numpy \
    fontconfig \
    fonts-dejavu \
    fonts-dejavu-core \
    fonts-dejavu-extra \
    # Health check and monitoring tools
    htop \
    iotop \
    jq \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install noVNC for browser viewing
RUN git clone https://github.com/novnc/noVNC.git /opt/novnc \
    && git clone https://github.com/novnc/websockify /opt/novnc/utils/websockify \
    && ln -s /opt/novnc/vnc.html /opt/novnc/index.html

# Create non-root user for security
RUN groupadd -r browseruse && useradd -r -g browseruse -s /bin/bash browseruse \
    && mkdir -p /home/browseruse \
    && chown -R browseruse:browseruse /home/browseruse

# Set up working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Install Playwright and browsers with system dependencies
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
RUN playwright install --with-deps chromium \
    && playwright install-deps

# Copy the application code
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p /app/data/chrome_data \
    /app/tmp/downloads \
    /app/tmp/traces \
    /app/tmp/recordings \
    /app/tmp/sessions \
    /app/logs \
    /var/log/supervisor \
    && chown -R browseruse:browseruse /app \
    && chown -R browseruse:browseruse /var/log/supervisor

# Set environment variables for production
ENV PYTHONUNBUFFERED=1 \
    BROWSER_USE_LOGGING_LEVEL=info \
    CHROME_PATH=/ms-playwright/chromium-*/chrome-linux/chrome \
    ANONYMIZED_TELEMETRY=false \
    DISPLAY=:99 \
    RESOLUTION=1920x1080x24 \
    VNC_PASSWORD=vncpassword \
    CHROME_PERSISTENT_SESSION=true \
    RESOLUTION_WIDTH=1920 \
    RESOLUTION_HEIGHT=1080 \
    # Multi-user configuration
    MAX_CONCURRENT_SESSIONS=20 \
    SESSION_TIMEOUT_MINUTES=45 \
    MAX_STEPS_PER_TASK=100 \
    # Server configuration
    HOST=0.0.0.0 \
    PORT=7788 \
    # Resource limits
    CHROME_DEBUGGING_PORT=9222 \
    CHROME_DEBUGGING_HOST=localhost

# Copy production supervisor configuration
COPY docker/supervisord.prod.conf /etc/supervisor/conf.d/supervisord.conf

# Copy health check script
COPY docker/healthcheck.sh /usr/local/bin/healthcheck.sh
RUN chmod +x /usr/local/bin/healthcheck.sh

# Copy entrypoint script
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Expose ports
EXPOSE 7788 6080 5901 9222

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD /usr/local/bin/healthcheck.sh

# Switch to non-root user
USER browseruse

# Set entrypoint
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

CMD ["supervisord"]
