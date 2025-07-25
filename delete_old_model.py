# delete_old_model.py
import shutil
import os

cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
old_model = "models--SnypzZz--Llama2-13b-Language-translate"
old_model_path = os.path.join(cache_dir, old_model)

if os.path.exists(old_model_path):
    shutil.rmtree(old_model_path)
    print(f"✅ 기존 모델 삭제됨: {old_model}")
else:
    print("기존 모델을 찾을 수 없습니다.")