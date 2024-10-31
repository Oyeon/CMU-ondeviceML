import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import time
from PIL import Image

# TensorRT Logger 설정
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(engine_file_path):
    # TensorRT 엔진 파일 로드
    with open(engine_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine

def allocate_buffers(engine):
    # 입력 및 출력 버퍼 할당
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        binding_shape = engine.get_binding_shape(binding)
        size = trt.volume(binding_shape)
        dtype = trt.nptype(engine.get_binding_dtype(binding))

        # 호스트 및 디바이스 메모리 할당
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        # 바인딩 정보 저장
        bindings.append(int(device_mem))

        if engine.binding_is_input(binding):
            inputs.append({'host': host_mem, 'device': device_mem})
        else:
            outputs.append({'host': host_mem, 'device': device_mem})

    return inputs, outputs, bindings, stream

def do_inference(context, bindings, inputs, outputs, stream):
    # 입력 데이터를 GPU로 복사
    [cuda.memcpy_htod_async(inp['device'], inp['host'], stream) for inp in inputs]
    # 추론 실행
    context.execute_async(bindings=bindings, stream_handle=stream.handle)
    # 결과를 GPU에서 호스트로 복사
    [cuda.memcpy_dtoh_async(out['host'], out['device'], stream) for out in outputs]
    # 스트림 동기화
    stream.synchronize()
    # 결과 반환
    return [out['host'] for out in outputs]

def run_inference_for_all_engines():
    # 엔진 파일 경로 설정
    engine_file_map = {
        'fp32': 'fp32_default.engine',
        'fp16': 'fp16_static.engine',
        'int8': 'int8_static.engine'
    }

    # 입력 데이터 준비
    image_path = 'datasets/val/images/002115.png'
    image = Image.open(image_path)
    image = image.resize((640, 640))  # 모델의 입력 크기에 맞게 조정
    input_data = np.array(image).astype(np.float32)
    input_data = np.transpose(input_data, (2, 0, 1))  # (H, W, C)에서 (C, H, W)로 변경
    input_data = np.expand_dims(input_data, axis=0)  # 배치 차원 추가

    for engine_type, engine_path in engine_file_map.items():
        # 엔진 로드
        engine = load_engine(engine_path)

        # 버퍼 할당
        inputs, outputs, bindings, stream = allocate_buffers(engine)

        # 컨텍스트 생성
        context = engine.create_execution_context()

        # 입력 버퍼에 데이터 복사
        np.copyto(inputs[0]['host'], input_data.ravel())

        # 추론 시간 측정
        times = []
        for _ in range(10):
            start_time = time.time()
            outputs_data = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            end_time = time.time()
            times.append(end_time - start_time)

        avg_time = sum(times) / len(times)
        print(f"[{engine_type.upper()}] 평균 추론 시간: {avg_time:.6f} 초")

if __name__ == "__main__":
    run_inference_for_all_engines()
