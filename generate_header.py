import tensorflow as tf
import config

model_path = config.TFLITE_PATH
with open(model_path, "rb") as f:
    tflite_quant_model = f.read()

header_path = config.HEADER_PATH
hex_array = ', '.join([f'0x{b:02x}' for b in tflite_quant_model])

header_content = f"""#ifndef MODEL_H_
#define MODEL_H_

#include <stdint.h>

alignas(16) const unsigned char model_tflite[] = {{
    {hex_array}
}};
const unsigned int model_tflite_len = {len(tflite_quant_model)};

#endif  // MODEL_H_
"""

with open(header_path, "w") as f:
    f.write(header_content)

print(f"Header successfully written to '{header_path}'. Ready for ESP-IDF/Arduino!")
