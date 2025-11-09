#!/bin/bash
# Quick test script for DeepSeek-OCR GCP service
# Usage: ./test_deepseek_gcp.sh YOUR_EXTERNAL_IP

set -e

ENDPOINT="${1:-http://localhost:8000}"

echo "Testing DeepSeek-OCR service at: ${ENDPOINT}"
echo ""

# Test 1: Health check
echo "1. Testing health endpoint..."
HEALTH_RESPONSE=$(curl -s "${ENDPOINT}/healthz")
echo "Response: ${HEALTH_RESPONSE}"
echo ""

# Verify service is healthy
if echo "${HEALTH_RESPONSE}" | grep -q "healthy"; then
    echo "✅ Service is healthy"
else
    echo "❌ Service health check failed"
    exit 1
fi

# Test 2: Create test image
echo "2. Creating test image..."
python3 << 'PYTHON_SCRIPT'
from PIL import Image
import base64
import io

# Create a simple test image with text-like content
img = Image.new('RGB', (400, 100), color='white')
img.save('test_image.png')
print("Test image created: test_image.png")

# Encode to base64
with open('test_image.png', 'rb') as f:
    img_base64 = base64.b64encode(f.read()).decode('utf-8')

# Save to file for use in curl
with open('test_image_base64.txt', 'w') as f:
    f.write(img_base64)

print("Base64 encoded image saved to: test_image_base64.txt")
PYTHON_SCRIPT

# Test 3: Validation request
echo ""
echo "3. Testing validation endpoint..."
VALIDATION_RESPONSE=$(curl -s -X POST "${ENDPOINT}/validate_piicrop" \
  -H "Content-Type: application/json" \
  -d "{
    \"image_base64\": \"$(cat test_image_base64.txt)\",
    \"prompt\": \"<image>\\n<|grounding|>Analyze this document region and determine if it contains a Social Insurance Number (SIN). Respond in JSON: {\\\"is_pii\\\": bool, \\\"subtype\\\": str, \\\"confidence\\\": 0.0-1.0, \\\"reason\\\": str}\",
    \"category\": \"SIN\",
    \"temperature\": 0.0,
    \"max_tokens\": 512
  }" \
  --max-time 60)

echo "Response: ${VALIDATION_RESPONSE}"
echo ""

# Check if response is valid
if echo "${VALIDATION_RESPONSE}" | grep -q "text\|output"; then
    echo "✅ Validation endpoint working"
else
    echo "❌ Validation endpoint failed"
    echo "Full response: ${VALIDATION_RESPONSE}"
    exit 1
fi

# Cleanup
rm -f test_image.png test_image_base64.txt

echo ""
echo "✅ All tests passed!"
echo "Service is ready to use."
echo ""
echo "Next steps:"
echo "1. Update docker-compose.env:"
echo "   DEEPSEEK_OCR_ENABLED=true"
echo "   DEEPSEEK_OCR_ENDPOINT=http://${ENDPOINT}"
echo "2. Restart pipeline: docker compose restart pst-processor"
echo "3. Monitor logs: docker compose logs -f pst-processor | grep 'L3 validation'"

