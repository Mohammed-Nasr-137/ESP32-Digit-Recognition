#include <stdio.h>
#include <string.h>
#include "model.h"
#include "esp_log.h"
#include "esp_wifi.h"
#include "esp_http_server.h"
#include "nvs_flash.h"
#include "esp_timer.h"

// TensorFlow Lite Micro API Includes
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {
    // Wi-Fi config
    #define WIFI_SSID      "ESP32_OCR_Network"
    #define WIFI_PASS      "digit123"
    #define WIFI_CHANNEL   1
    #define MAX_STA_CONN   4

    // Logging tag for monitor
    const char* TAG = "OCR_INF";

    // runtime pointers
    const tflite::Model* model = nullptr;
    tflite::MicroInterpreter* interpreter = nullptr;
    TfLiteTensor* input = nullptr;
    TfLiteTensor* output = nullptr;

    // size was determined using interpreter->arena_used_bytes()
    constexpr int kTensorArenaSize = 3100;
    alignas(16) uint8_t tensor_arena[kTensorArenaSize];
}

const char* index_html = R"rawhtml(
<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ESP32 16x16 Digit OCR</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; text-align: center; margin: 15px; background: #121212; color: #eee; }
        .container { display: flex; flex-direction: column; align-items: center; gap: 12px; }
        #viewfinder { width: 260px; height: 260px; border: 3px solid #444; border-radius: 8px; background: #000; object-fit: cover; }
        #processing-canvas { 
            width: 128px; 
            height: 128px; 
            border: 2px dashed #00e676; 
            background: #000;
            image-rendering: pixelated; 
            image-rendering: crisp-edges;
            border-radius: 4px;
        }
        button { padding: 14px 28px; font-size: 16px; font-weight: bold; cursor: pointer; background: #2979ff; color: white; border: none; border-radius: 6px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
        button:active { background: #1565c0; transform: translateY(2px); }
        #result { font-size: 28px; font-weight: bold; margin-top: 5px; color: #00e676; }
        #metrics { font-size: 14px; color: #aaa; font-family: monospace; }
        .label-text { font-size: 13px; color: #bbb; }
    </style>
</head>
<body>
    <h2>ESP32 Edge Digit OCR</h2>
    <div class="container">
        <video id="viewfinder" autoplay playsinline></video>
        <div class="label-text">Point camera at a single dark handwritten digit</div>
        
        <button id="capture-btn">Recognize Digit</button>
        
        <canvas id="processing-canvas" width="16" height="16"></canvas>
        <div class="label-text">Preprocessed 16x16 Tensor Input</div>
        
        <div id="result">Prediction: --</div>
        <div id="metrics">Inference Latency: -- ms</div>
    </div>

    <script>
    const video = document.getElementById('viewfinder');
    const canvas16Display = document.getElementById('processing-canvas');
    const ctx16 = canvas16Display.getContext('2d');
    const resultDiv = document.getElementById('result');
    const metricsDiv = document.getElementById('metrics');

    let stream = null;

    async function startCamera() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            resultDiv.innerText = 'Camera API not supported';
            return;
        }
        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: 'environment', width: 280, height: 280 }
            });
            video.srcObject = stream;
            await video.play();
        } catch (err) {
            resultDiv.innerText = 'Camera access denied';
            console.error(err);
        }
    }

    // Adaptive Thresholding via O(1) Integral Images + 3x3 Morphology + Bounding Box + 12x12 Resize
    function processFrame(videoElement, captureDim = 140) {
        const captureCanvas = document.createElement('canvas');
        captureCanvas.width = captureDim;
        captureCanvas.height = captureDim;
        const ctx = captureCanvas.getContext('2d', { willReadFrequently: true });
        ctx.drawImage(videoElement, 0, 0, captureDim, captureDim);

        const imgData = ctx.getImageData(0, 0, captureDim, captureDim);
        const pixels = imgData.data;

        // 1. Grayscale Conversion
        const gray = new Uint8Array(captureDim * captureDim);
        for (let i = 0, j = 0; i < pixels.length; i += 4, j++) {
            gray[j] = Math.round(0.299 * pixels[i] + 0.587 * pixels[i + 1] + 0.114 * pixels[i + 2]);
        }

        // 2. Build Integral Image for Fast Local Mean Calculation
        const S = captureDim;
        const integral = new Int32Array((S + 1) * (S + 1));
        for (let y = 0; y < S; y++) {
            let rowSum = 0;
            for (let x = 0; x < S; x++) {
                rowSum += gray[y * S + x];
                integral[(y + 1) * (S + 1) + (x + 1)] = integral[y * (S + 1) + (x + 1)] + rowSum;
            }
        }

        // 3. Local Adaptive Thresholding (Window: 21x21, Offset C: 8, Margin: 8px)
        const binary = new Uint8Array(S * S);
        const R = 10;     // Neighborhood radius (21x21 window)
        const C = 8;      // Constant threshold offset
        const margin = 8; // Ignore outermost 8px to eliminate camera edge vignetting

        for (let y = 0; y < S; y++) {
            for (let x = 0; x < S; x++) {
                if (x < margin || x >= S - margin || y < margin || y >= S - margin) {
                    binary[y * S + x] = 0;
                    continue;
                }

                const x0 = Math.max(0, x - R);
                const x1 = Math.min(S, x + R + 1);
                const y0 = Math.max(0, y - R);
                const y1 = Math.min(S, y + R + 1);

                const count = (x1 - x0) * (y1 - y0);
                const sum = integral[y1 * (S + 1) + x1] 
                          - integral[y0 * (S + 1) + x1] 
                          - integral[y1 * (S + 1) + x0] 
                          + integral[y0 * (S + 1) + x0];

                const localMean = sum / count;
                binary[y * S + x] = (gray[y * S + x] < (localMean - C)) ? 255 : 0;
            }
        }

        // 4. 3x3 Morphology Functions
        function dilate3x3(src) {
            const dst = new Uint8Array(S * S);
            for (let y = 0; y < S; y++) {
                for (let x = 0; x < S; x++) {
                    let hit = 0;
                    for (let dy = -1; dy <= 1; dy++) {
                        for (let dx = -1; dx <= 1; dx++) {
                            const ny = y + dy, nx = x + dx;
                            if (nx >= 0 && nx < S && ny >= 0 && ny < S) {
                                if (src[ny * S + nx] === 255) { hit = 255; break; }
                            }
                        }
                        if (hit === 255) break;
                    }
                    dst[y * S + x] = hit;
                }
            }
            return dst;
        }

        function erode3x3(src) {
            const dst = new Uint8Array(S * S);
            for (let y = 0; y < S; y++) {
                for (let x = 0; x < S; x++) {
                    let allOn = 255;
                    for (let dy = -1; dy <= 1; dy++) {
                        for (let dx = -1; dx <= 1; dx++) {
                            const ny = y + dy, nx = x + dx;
                            if (nx >= 0 && nx < S && ny >= 0 && ny < S) {
                                if (src[ny * S + nx] !== 255) { allOn = 0; break; }
                            } else { allOn = 0; break; }
                        }
                        if (allOn === 0) break;
                    }
                    dst[y * S + x] = allOn;
                }
            }
            return dst;
        }

        // Morphological Pipeline: Close -> Dilate
        const morphed = dilate3x3(erode3x3(dilate3x3(binary)));

        // 5. Extract Bounding Box
        let minX = S, maxX = 0, minY = S, maxY = 0, hasInk = false;
        for (let y = 0; y < S; y++) {
            for (let x = 0; x < S; x++) {
                if (morphed[y * S + x] === 255) {
                    if (x < minX) minX = x;
                    if (x > maxX) maxX = x;
                    if (y < minY) minY = y;
                    if (y > maxY) maxY = y;
                    hasInk = true;
                }
            }
        }

        // Filter out accidental tiny noise specks (< 8px total size)
        if (!hasInk || (maxX - minX < 8 && maxY - minY < 8)) return null;

        const bboxW = maxX - minX + 1;
        const bboxH = maxY - minY + 1;

        // 6. Draw Morphed Mask to Offscreen Canvas for Area Downsampling
        const maskCanvas = document.createElement('canvas');
        maskCanvas.width = S;
        maskCanvas.height = S;
        const maskCtx = maskCanvas.getContext('2d');
        const maskImgData = maskCtx.createImageData(S, S);
        for (let i = 0; i < morphed.length; i++) {
            const val = morphed[i];
            const p = i * 4;
            maskImgData.data[p] = val;
            maskImgData.data[p + 1] = val;
            maskImgData.data[p + 2] = val;
            maskImgData.data[p + 3] = 255;
        }
        maskCtx.putImageData(maskImgData, 0, 0);

        // 7. Aspect-Preserving Scaling (Fit within 12x12)
        const scale = Math.min(12.0 / bboxW, 12.0 / bboxH);
        const newW = Math.max(1, Math.round(bboxW * scale));
        const newH = Math.max(1, Math.round(bboxH * scale));

        const scaledCanvas = document.createElement('canvas');
        scaledCanvas.width = newW;
        scaledCanvas.height = newH;
        const scaledCtx = scaledCanvas.getContext('2d');
        scaledCtx.imageSmoothingEnabled = true;
        scaledCtx.imageSmoothingQuality = 'high';
        scaledCtx.drawImage(maskCanvas, minX, minY, bboxW, bboxH, 0, 0, newW, newH);

        const scaledPixels = scaledCtx.getImageData(0, 0, newW, newH).data;

        // 8. Place Centered in 16x16 Flat Buffer
        const payload256 = new Uint8Array(256).fill(0);
        const startX = Math.floor((16 - newW) / 2);
        const startY = Math.floor((16 - newH) / 2);

        for (let y = 0; y < newH; y++) {
            for (let x = 0; x < newW; x++) {
                const srcIdx = (y * newW + x) * 4;
                const destIdx = (startY + y) * 16 + (startX + x);
                payload256[destIdx] = scaledPixels[srcIdx];
            }
        }

        return payload256;
    }

    document.getElementById('capture-btn').addEventListener('click', async () => {
        if (!stream) await startCamera();
        if (!stream || video.readyState < 2) return;

        resultDiv.innerText = 'Analyzing...';

        const payload256 = processFrame(video, 140);
        if (!payload256) {
            resultDiv.innerText = 'No Digit Detected';
            return;
        }

        // Render live 16x16 input to verify stroke extraction
        const displayData = ctx16.createImageData(16, 16);
        for (let i = 0; i < 256; i++) {
            const val = payload256[i];
            const p = i * 4;
            displayData.data[p] = val;
            displayData.data[p + 1] = val;
            displayData.data[p + 2] = val;
            displayData.data[p + 3] = 255;
        }
        ctx16.putImageData(displayData, 0, 0);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/octet-stream' },
                body: payload256
            });
            const data = await response.json();
            resultDiv.innerText = 'Prediction: ' + data.digit;
            metricsDiv.innerText = 'Inference Latency: ' + data.inference_time_ms.toFixed(2) + ' ms';
        } catch (err) {
            resultDiv.innerText = 'Inference Failed';
            console.error(err);
        }
    });

    startCamera();
</script>
</body>
</html>
)rawhtml";


// Inference Engine setup
void setup_model() {
    // map model
    model = tflite::GetModel(model_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Schema mismatch! Model version %d != Runtime version %d", 
                 model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }

    // register operators used by model
    static tflite::MicroMutableOpResolver<5> micro_op_resolver;
    if (micro_op_resolver.AddConv2D() != kTfLiteOk) return;
    if (micro_op_resolver.AddMaxPool2D() != kTfLiteOk) return;
    if (micro_op_resolver.AddReshape() != kTfLiteOk) return;
    if (micro_op_resolver.AddFullyConnected() != kTfLiteOk) return;
    if (micro_op_resolver.AddTranspose() != kTfLiteOk) return;

    // init the interpreter
    static tflite::MicroInterpreter static_interpreter(
        model, micro_op_resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    // allocate tensor activation buffers in the arena
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors() failed! Increase kTensorArenaSize.");
        return;
    }

    // measure exact memory req
    size_t actual_used_bytes = interpreter->arena_used_bytes();
    ESP_LOGW(TAG, "==================================================");
    ESP_LOGW(TAG, "CALIBRATION RESULT: Exact Arena Needed = %d bytes", actual_used_bytes);
    ESP_LOGW(TAG, "==================================================");

    input = interpreter->input(0);
    output = interpreter->output(0);
}

// HTTP server handlers

// GET / -> Serves embedded web page
esp_err_t root_get_handler(httpd_req_t *req) {
    httpd_resp_set_type(req, "text/html");
    httpd_resp_send(req, index_html, HTTPD_RESP_USE_STRLEN);
    return ESP_OK;
}

// POST /predict -> Ingests 256-byte payload, runs inference, returns JSON
esp_err_t predict_post_handler(httpd_req_t *req) {
    // check if invalid size
    if (req->content_len != 256) {
        ESP_LOGE(TAG, "Invalid payload length: %d bytes (Expected: 256)", req->content_len);
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Payload must be 256 bytes");
        return ESP_FAIL;
    }

    // read tcp socket buffer into model input tensor memory
    int remaining = req->content_len;
    int bytes_read;
    while (remaining > 0) {
        bytes_read = httpd_req_recv(req, (char*)(input->data.int8 + (req->content_len - remaining)), remaining);
        if (bytes_read <= 0) {
            if (bytes_read == HTTPD_SOCK_ERR_TIMEOUT) continue;
            ESP_LOGE(TAG, "Socket receive error: %d", bytes_read);
            httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "Socket error");
            return ESP_FAIL;
        }
        remaining -= bytes_read;
    }

    // map [0..255] uint8 -> TFLM [-128..127] int8
    for (int i = 0; i < 256; i++) {
        uint8_t raw_pixel = (uint8_t)input->data.int8[i];
        input->data.int8[i] = (int16_t)raw_pixel - 128;
    }

    // run inference
    int64_t start_time = esp_timer_get_time();
    if (interpreter->Invoke() != kTfLiteOk) {
        ESP_LOGE(TAG, "interpreter->Invoke() failed!");
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "Inference error");
        return ESP_FAIL;
    }
    int64_t end_time = esp_timer_get_time();
    float latency_ms = (float)(end_time - start_time) / 1000.0f;

    // argmax
    int8_t max_logit = -128;
    int predicted_digit = 0;
    for (int i = 0; i < 10; i++) {
        if (output->data.int8[i] > max_logit) {
            max_logit = output->data.int8[i];
            predicted_digit = i;
        }
    }

    char response_json[128];
    snprintf(response_json, sizeof(response_json), 
             "{\"digit\": %d, \"confidence_raw\": %d, \"inference_time_ms\": %.2f}", 
             predicted_digit, max_logit, latency_ms);

    httpd_resp_set_type(req, "application/json");
    httpd_resp_send(req, response_json, HTTPD_RESP_USE_STRLEN);

    ESP_LOGI(TAG, "Predicted: %d | Time: %.2f ms | Raw Logit: %d", predicted_digit, latency_ms, max_logit);
    return ESP_OK;
}

// HTTP server launcher
httpd_handle_t start_webserver() {
    httpd_handle_t server = nullptr;
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.lru_purge_enable = true;

    if (httpd_start(&server, &config) == ESP_OK) {
        // Register GET /
        httpd_uri_t root_uri = {
            .uri       = "/",
            .method    = HTTP_GET,
            .handler   = root_get_handler,
            .user_ctx  = nullptr
        };
        httpd_register_uri_handler(server, &root_uri);

        // Register POST /predict
        httpd_uri_t predict_uri = {
            .uri       = "/predict",
            .method    = HTTP_POST,
            .handler   = predict_post_handler,
            .user_ctx  = nullptr
        };
        httpd_register_uri_handler(server, &predict_uri);

        ESP_LOGI(TAG, "HTTP Daemon mounted on port 80.");
        return server;
    }
    return nullptr;
}


// Wi-Fi Event Handler Callback
static void wifi_event_handler(void* arg, esp_event_base_t event_base, int32_t event_id, void* event_data) {
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_AP_STACONNECTED) {
        auto* event = (wifi_event_ap_staconnected_t*) event_data;
        ESP_LOGI(TAG, "Station connected | AID: %d", event->aid);
    }
    else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_AP_STADISCONNECTED) {
        auto* event = (wifi_event_ap_stadisconnected_t*) event_data;
        ESP_LOGW(TAG, "Station disconnected");
    }
}

// Wi-Fi AP init
void wifi_init_ap() {
    // init tcp/ip stack
    ESP_ERROR_CHECK(esp_netif_init());

    // create default system loop task
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    // create network interface instance for ap
    esp_netif_t* ap_netif = esp_netif_create_default_wifi_ap();

    // init Wi-Fi driver with default system allocations
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    // register Wi-Fi event handler to monitor connections
    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        WIFI_EVENT,
        ESP_EVENT_ANY_ID,
        &wifi_event_handler,
        nullptr,
        nullptr
    ));

    // configure AP paramteres
    wifi_config_t wifi_config = {};
    strncpy((char*)wifi_config.ap.ssid, WIFI_SSID, sizeof(wifi_config.ap.ssid));
    wifi_config.ap.ssid_len = strlen(WIFI_SSID);
    strncpy((char*)wifi_config.ap.password, WIFI_PASS, sizeof(wifi_config.ap.password));
    wifi_config.ap.channel = WIFI_CHANNEL;
    wifi_config.ap.max_connection = MAX_STA_CONN;
    wifi_config.ap.authmode = WIFI_AUTH_WPA_WPA2_PSK;

    // commit config and launch AP
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_AP));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_AP, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "==================================================");
    ESP_LOGI(TAG, "AP Active! SSID: \"%s\" | Password: \"%s\"", WIFI_SSID, WIFI_PASS);
    ESP_LOGI(TAG, "==================================================");
}

extern "C" void app_main(void)
{
    // Initialize NVS
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    setup_model();
    wifi_init_ap();
    start_webserver();
}
