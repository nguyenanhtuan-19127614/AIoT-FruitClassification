#include <a19127632_human_face_detection_inferencing.h>

#include "esp_timer.h"
#include "esp_camera.h"
// #include "img_converters.h"
// #include "image_util.h"
//#include "camera_index_html.h"
#include "Arduino.h"

#include "fb_gfx.h"
#include "fd_forward.h"
//#include "dl_lib.h"
#include "fr_forward.h"

/* Modify the following line according to your project name
   Do not forget to import the library using "Sketch">"Include Library">"Add .ZIP Library..."
*/

uint8_t *out_buf;
uint8_t *ei_buf;

static int8_t ei_activate = 0;
dl_matrix3du_t *resized_matrix = NULL;
size_t out_len = EI_CLASSIFIER_INPUT_WIDTH * EI_CLASSIFIER_INPUT_HEIGHT;
ei_impulse_result_t result = {0};




int raw_feature_get_data(size_t offset, size_t out_len, float *signal_ptr)
{
  size_t pixel_ix = offset * 3;
  size_t bytes_left = out_len;
  size_t out_ptr_ix = 0;

  // read byte for byte
  while (bytes_left != 0) {
    // grab the values and convert to r/g/b
    uint8_t r, g, b;
    r = ei_buf[pixel_ix];
    g = ei_buf[pixel_ix + 1];
    b = ei_buf[pixel_ix + 2];

    // then convert to out_ptr format
    float pixel_f = (r << 16) + (g << 8) + b;
    signal_ptr[out_ptr_ix] = pixel_f;

    // and go to the next pixel
    out_ptr_ix++;
    pixel_ix += 3;
    bytes_left--;
  }
  return 0;
}

void classify()
{
  Serial.println("Getting signal...");
  signal_t signal;
  signal.total_length = EI_CLASSIFIER_INPUT_WIDTH * EI_CLASSIFIER_INPUT_WIDTH;
  signal.get_data = &raw_feature_get_data;

  Serial.println("Run classifier...");
  // Feed signal to the classifier
  EI_IMPULSE_ERROR res = run_classifier(&signal, &result, false /* debug */);

  // Returned error variable "res" while data object.array in "result"
  ei_printf("run_classifier returned: %d\n", res);
  if (res != 0)
    return;

  // print the predictions
  ei_printf("Predictions (DSP: %d ms., Classification: %d ms., Anomaly: %d ms.): \n",
            result.timing.dsp, result.timing.classification, result.timing.anomaly);
  for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
    ei_printf("    %s: \t%f\r\n", result.classification[ix].label, result.classification[ix].value);
  }
#if EI_CLASSIFIER_HAS_ANOMALY == 1
  ei_printf("    anomaly score: %f\r\n", result.anomaly);
#endif
}

esp_err_t inference_handler()
{
  camera_fb_t *fb = NULL;
  esp_err_t res = ESP_OK;

  size_t out_len, out_width, out_height;
  size_t ei_len;

  fb = esp_camera_fb_get();
  if (!fb)
  {
    Serial.println("Camera capture failed");
    return ESP_FAIL;
  }

  bool s;
  bool detected = false;

  dl_matrix3du_t *image_matrix = dl_matrix3du_alloc(1, fb->width, fb->height, 3);
  if (!image_matrix)
  {
    esp_camera_fb_return(fb);
    Serial.println("dl_matrix3du_alloc failed");
    return ESP_FAIL;
  }

  out_buf = image_matrix->item;
  out_len = fb->width * fb->height * 3;
  out_width = fb->width;
  out_height = fb->height;

  Serial.println("Converting to RGB888...");
  int64_t time_start = esp_timer_get_time();
  s = fmt2rgb888(fb->buf, fb->len, fb->format, out_buf);
  int64_t time_end = esp_timer_get_time();
  Serial.printf("Done in %ums\n", (uint32_t)((time_end - time_start) / 1000));

  esp_camera_fb_return(fb);
  if (!s)
  {
    dl_matrix3du_free(image_matrix);
    Serial.println("to rgb888 failed");
    return ESP_FAIL;
  }

  dl_matrix3du_t *ei_matrix = dl_matrix3du_alloc(1, EI_CLASSIFIER_INPUT_WIDTH, EI_CLASSIFIER_INPUT_HEIGHT, 3);
  if (!ei_matrix)
  {
    esp_camera_fb_return(fb);
    return ESP_FAIL;
  }

  ei_buf = ei_matrix->item;
  ei_len = EI_CLASSIFIER_INPUT_WIDTH * EI_CLASSIFIER_INPUT_HEIGHT * 3;

  Serial.println("Resizing the frame buffer...");
  time_start = esp_timer_get_time();
  image_resize_linear(ei_buf, out_buf, EI_CLASSIFIER_INPUT_WIDTH, EI_CLASSIFIER_INPUT_HEIGHT, 3, out_width, out_height);
  time_end = esp_timer_get_time();
  Serial.printf("Done in %ums\n", (uint32_t)((time_end - time_start) / 1000));

  dl_matrix3du_free(image_matrix);

  classify();

  dl_matrix3du_free(ei_matrix);
  return res;
}
