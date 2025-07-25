#!/usr/bin/env python
"""
대체 번역 방법 테스트
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import torch

def test_direct_translation():
    """직접 번역 테스트"""
    print("직접 번역 테스트...")
    print("=" * 50)
    
    # 모델과 토크나이저 로드
    print("모델 로드 중...")
    model = MBartForConditionalGeneration.from_pretrained("SnypzZz/Llama2-13b-Language-translate")
    tokenizer = MBart50TokenizerFast.from_pretrained("SnypzZz/Llama2-13b-Language-translate")
    
    # 테스트 케이스
    test_cases = [
        {
            "text": "안녕하세요, 반갑습니다.",
            "src_lang": "ko_KR",
            "tgt_lang": "en_XX"
        },
        {
            "text": "Hello, nice to meet you.",
            "src_lang": "en_XX", 
            "tgt_lang": "ko_KR"
        },
        {
            "text": "안녕하세요.",
            "src_lang": "ko_KR",
            "tgt_lang": "ja_XX"
        }
    ]
    
    for i, test in enumerate(test_cases):
        print(f"\n테스트 {i+1}: {test['src_lang']} → {test['tgt_lang']}")
        print(f"원문: {test['text']}")
        
        try:
            # 소스 언어 설정
            tokenizer.src_lang = test['src_lang']
            
            # 텍스트 인코딩
            encoded = tokenizer(test['text'], return_tensors="pt", padding=True)
            
            # 생성 파라미터 설정
            forced_bos_token_id = tokenizer.lang_code_to_id[test['tgt_lang']]
            
            # 번역 생성
            generated_tokens = model.generate(
                **encoded,
                forced_bos_token_id=forced_bos_token_id,
                max_length=100,
                num_beams=5,
                num_return_sequences=1,
                temperature=1.0,
                do_sample=False,
                top_k=50,
                top_p=0.95
            )
            
            # 디코딩
            translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            print(f"번역: {translation[0]}")
            
        except Exception as e:
            print(f"오류: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_direct_translation()