# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
import os
import sys
import subprocess
import numpy as np
import onnxruntime as ort

def get_apu_info():
    # Run pnputil as a subprocess to enumerate PCI devices
    command = r'pnputil /enum-devices /bus PCI /deviceids '
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    
    # Check for supported Hardware IDs - use 'cp1252' encoding with error handling
    try:
        # Try with the Windows default encoding
        stdout_text = stdout.decode('cp1252', errors='replace')
    except:
        # Fallback to a safer approach that will at least not crash
        stdout_text = str(stdout)
    
    apu_type = ''
    if 'PCI\\VEN_1022&DEV_1502&REV_00' in stdout_text: apu_type = 'PHX/HPT'
    if 'PCI\\VEN_1022&DEV_17F0&REV_00' in stdout_text: apu_type = 'STX'
    if 'PCI\\VEN_1022&DEV_17F0&REV_10' in stdout_text: apu_type = 'STX'
    if 'PCI\\VEN_1022&DEV_17F0&REV_11' in stdout_text: apu_type = 'STX'
    if 'PCI\\VEN_1022&DEV_17F0&REV_20' in stdout_text: apu_type = 'KRK'
    return apu_type

def set_environment_variable(apu_type):
    install_dir = os.environ['RYZEN_AI_INSTALLATION_PATH']
    os.environ['XLNX_ENABLE_CACHE']='0'
    match apu_type:
        case 'PHX/HPT':
            print("Setting environment for PHX/HPT")
            os.environ['XLNX_VART_FIRMWARE']= os.path.join(install_dir, 'voe-4.0-win_amd64', 'xclbins', 'phoenix', '1x4.xclbin')
            os.environ['NUM_OF_DPU_RUNNERS']='1'
            os.environ['XLNX_TARGET_NAME']='AMD_AIE2_Nx4_Overlay'
        case ('STX' | 'KRK'):
            print("Setting environment for STX")
            os.environ['XLNX_VART_FIRMWARE']= os.path.join(install_dir, 'voe-4.0-win_amd64', 'xclbins', 'strix', 'AMD_AIE2P_Nx4_Overlay.xclbin')
            os.environ['NUM_OF_DPU_RUNNERS']='1'
            os.environ['XLNX_TARGET_NAME']='AMD_AIE2_Nx4_Overlay'
        case _:
            print("Unrecognized APU type. Exiting.")
            exit()
    print('XLNX_VART_FIRMWARE=', os.environ['XLNX_VART_FIRMWARE'])
    print('NUM_OF_DPU_RUNNERS=', os.environ['NUM_OF_DPU_RUNNERS'])
    print('XLNX_TARGET_NAME=', os.environ['XLNX_TARGET_NAME'])
    print('XLNX_ENABLE_CACHE=', os.environ['XLNX_ENABLE_CACHE'])

# Get APU type info: PHX/STX/HPT
apu_type = get_apu_info()

# set environment variables: XLNX_VART_FIRMWARE and NUM_OF_DPU_RUNNERS
set_environment_variable(apu_type)

install_dir = os.environ['RYZEN_AI_INSTALLATION_PATH']
model       = os.path.join(install_dir, 'quicktest', 'test_model.onnx')
config_file = os.path.join(install_dir, 'voe-4.0-win_amd64', 'vaip_config.json')
cache_key   = 'modelcachekey_quicktest'
providers   = ['VitisAIExecutionProvider']
provider_options = [{
            'config_file': config_file,
            'cacheKey': cache_key,
        }]

try:
    session = ort.InferenceSession(model, providers=providers,
                               provider_options=provider_options)
except Exception as e:
    print(f"Failed to create an InferenceSession: {e}")
    sys.exit(1)  # Exit the program with a non-zero status to indicate an error

def preprocess_random_image():
    image_array = np.random.rand(3, 32, 32).astype(np.float32)
    return np.expand_dims(image_array, axis=0)

# inference on random image data
input_data = preprocess_random_image()
try:
    outputs = session.run(None, {'input': input_data})
except Exception as e:
    print(f"Failed to run the InferenceSession: {e}")
    sys.exit(1)  # Exit the program with a non-zero status to indicate an error
else:
    print("Test Passed")
