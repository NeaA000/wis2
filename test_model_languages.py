#!/usr/bin/env python
"""
모델이 지원하는 언어 확인
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import MBart50TokenizerFast

def check_supported_languages():
    """지원 언어 확인"""
    print("모델 언어 지원 확인...")
    print("=" * 50)
    
    try:
        # 토크나이저 로드
        tokenizer = MBart50TokenizerFast.from_pretrained("SnypzZz/Llama2-13b-Language-translate")
        
        print("지원하는 언어 코드:")
        print("-" * 30)
        
        # lang_code_to_id 확인
        if hasattr(tokenizer, 'lang_code_to_id'):
            for lang_code, lang_id in sorted(tokenizer.lang_code_to_id.items()):
                print(f"{lang_code}: {lang_id}")
        else:
            print("lang_code_to_id 속성을 찾을 수 없습니다.")
            
        print("\n추가 정보:")
        print(f"기본 소스 언어: {tokenizer.src_lang if hasattr(tokenizer, 'src_lang') else 'N/A'}")
        print(f"토크나이저 타입: {type(tokenizer)}")
        
        # 한국어 토큰 테스트
        print("\n한국어 토큰화 테스트:")
        test_text = "안녕하세요"
        tokens = tokenizer.tokenize(test_text)
        print(f"텍스트: {test_text}")
        print(f"토큰: {tokens}")
        
        # 다른 방법으로 번역 테스트
        print("\n대체 번역 방법 테스트:")
        tokenizer.src_lang = "ko_KR"
        inputs = tokenizer(test_text, return_tensors="pt")
        print(f"입력 ID 길이: {inputs['input_ids'].shape}")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_supported_languages()