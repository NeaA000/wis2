#!/usr/bin/env python
"""
번역 기능 독립 테스트 스크립트
"""
import os
import sys

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from auto_subtitle_llama.cli import translates
from auto_subtitle_llama.utils import WHISPER_TO_MBART_LANG_CODE

def test_translation():
    """번역 테스트"""
    # 테스트 케이스들
    test_cases = [
        {
            "source_lang": "ko_KR",
            "target_lang": "en_XX",
            "texts": [
                "안녕하세요, 반갑습니다.",
                "오늘 날씨가 정말 좋네요.",
                "이 프로그램은 자동 번역 기능을 제공합니다."
            ]
        },
        {
            "source_lang": "en_XX",
            "target_lang": "es_XX",
            "texts": [
                "Hello, nice to meet you.",
                "The weather is really nice today.",
                "This program provides automatic translation."
            ]
        },
        {
            "source_lang": "ko_KR",
            "target_lang": "ja_XX",
            "texts": [
                "안녕하세요.",
                "감사합니다.",
                "좋은 하루 보내세요."
            ]
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\n테스트 {i+1}: {test_case['source_lang']} → {test_case['target_lang']}")
        print(f"원문: {test_case['texts']}")
        print("-" * 50)
        
        try:
            translated = translates(
                translate_to=test_case['target_lang'],
                text_batch=test_case['texts'],
                source_lang=test_case['source_lang']
            )
            
            print(f"번역 결과:")
            for j, (original, translated_text) in enumerate(zip(test_case['texts'], translated)):
                print(f"{j+1}. {original}")
                print(f"   → {translated_text}")
                print()
                
        except Exception as e:
            print(f"번역 오류: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    print("번역 기능 테스트 시작...")
    print("=" * 50)
    test_translation()
    print("=" * 50)
    print("테스트 완료")