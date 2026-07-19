import nncase
import numpy as np
import os

# 校准数据: 20 张随机图
calib = [np.random.randint(0, 256, (1, 3, 640, 640)).astype(np.float32) for _ in range(20)]

# 编译选项
co = nncase.CompileOptions()
co.target = "k230"
co.input_type = "uint8"
co.input_shape = [1, 3, 640, 640]
co.input_mean = [0.0, 0.0, 0.0]
co.input_std = [255.0, 255.0, 255.0]
co.input_range = [0.0, 255.0]
co.preprocess = True
co.quant_type = "uint8"
co.calibration_dataset = calib

# 导入 ONNX
c = nncase.Compiler(co)
io = nncase.ImportOptions()
io.input_shape = [1, 3, 640, 640]
c.import_onnx("/home/aliuex/best.onnx", io)

# 编译
print("[INFO] Compiling...")
c.compile()
print("[INFO] Done!")

# 保存
output = "/home/aliuex/best.kmodel"
with open(output, "wb") as f:
    f.write(c.gencode_tobytes())

size_kb = os.path.getsize(output) / 1024
print(f"[SUCCESS] kmodel saved! Size: {size_kb:.1f} KB")
