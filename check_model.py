#!/usr/bin/env python
"""
현재 사용 중인 모델 정보 확인
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoModel, AutoTokenizer
import torch

def check_model_info():
    """모델 정보 확인"""
    model_name = "SnypzZz/Llama2-13b-Language-translate"
    
    print(f"모델 확인: {model_name}")
    print("=" * 60)
    
    try:
        # 1. 모델 config 확인
        print("\n1. 모델 설정 파일 확인:")
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name)
        print(f"- 모델 타입: {config.model_type}")
        print(f"- 아키텍처: {config.architectures}")
        print(f"- 숨겨진 크기: {getattr(config, 'hidden_size', 'N/A')}")
        print(f"- 레이어 수: {getattr(config, 'num_hidden_layers', 'N/A')}")
        
        # 2. 토크나이저 확인
        print("\n2. 토크나이저 확인:")
        try:
            from transformers import MBart50TokenizerFast
            tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
            print(f"- MBart50TokenizerFast: ✅ 로드 성공")
            print(f"- 토크나이저 클래스: {type(tokenizer)}")
            print(f"- 어휘 크기: {tokenizer.vocab_size}")
        except Exception as e:
            print(f"- MBart50TokenizerFast: ❌ 로드 실패 - {e}")
            
        try:
            from transformers import LlamaTokenizer
            tokenizer = LlamaTokenizer.from_pretrained(model_name)
            print(f"- LlamaTokenizer: ✅ 로드 성공")
            print(f"- 토크나이저 클래스: {type(tokenizer)}")
        except Exception as e:
            print(f"- LlamaTokenizer: ❌ 로드 실패 - {e}")
        
        # 3. 모델 로드 시도
        print("\n3. 모델 로드 시도:")
        
        # MBart로 로드
        try:
            from transformers import MBartForConditionalGeneration
            model = MBartForConditionalGeneration.from_pretrained(
                model_name, 
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            print(f"- MBartForConditionalGeneration: ✅ 로드 성공")
            print(f"- 모델 클래스: {type(model)}")
            
            # 모델 파라미터 정보
            total_params = sum(p.numel() for p in model.parameters())
            print(f"- 총 파라미터 수: {total_params:,}")
            
            del model  # 메모리 해제
        except Exception as e:
            print(f"- MBartForConditionalGeneration: ❌ 로드 실패 - {e}")
        
        # LLaMA로 로드
        try:
            from transformers import LlamaForCausalLM
            model = LlamaForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            print(f"- LlamaForCausalLM: ✅ 로드 성공")
            print(f"- 모델 클래스: {type(model)}")
            del model
        except Exception as e:
            print(f"- LlamaForCausalLM: ❌ 로드 실패 - {e}")
            
        # 4. 자동 감지
        print("\n4. AutoModel로 자동 감지:")
        try:
            model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            print(f"- 자동 감지된 모델 타입: {type(model)}")
            print(f"- 모델 구조의 첫 부분:\n{str(model)[:500]}...")
            del model
        except Exception as e:
            print(f"- AutoModel 로드 실패: {e}")
            
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_model_info()