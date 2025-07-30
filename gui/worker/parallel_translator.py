"""
병렬 번역 처리를 위한 클래스
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from typing import List, Dict, Tuple
import threading
import time

class ParallelTranslator:
    """언어별 병렬 번역 처리"""
    
    def __init__(self, worker, max_workers=3):
        self.worker = worker
        self.max_workers = max_workers
        self.translation_progress = {}
        self.lock = threading.Lock()
        
    def translate_multiple_languages(self, 
                                   segments: List[dict], 
                                   target_languages: List[str],
                                   source_lang: str,
                                   video_name: str) -> Dict[str, List[dict]]:
        """여러 언어로 동시 번역"""
        
        # 번역 모델 가져오기
        from auto_subtitle_llama.utils import TranslatorManager
        manager = TranslatorManager()
        model, tokenizer = manager.get_translator()
        
        # 진행률 초기화
        for lang in target_languages:
            self.translation_progress[lang] = 0
            
        results = {}
        
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(target_languages))) as executor:
            # 각 언어별로 번역 작업 제출
            future_to_lang = {}
            
            for target_lang in target_languages:
                # 같은 언어는 건너뛰기
                if source_lang.startswith(target_lang.split('_')[0]):
                    continue
                    
                future = executor.submit(
                    self._translate_single_language,
                    segments, target_lang, source_lang, 
                    model, tokenizer, video_name
                )
                future_to_lang[future] = target_lang
            
            # 완료된 것부터 처리
            for future in as_completed(future_to_lang):
                target_lang = future_to_lang[future]
                try:
                    translated_segments = future.result()
                    results[target_lang] = translated_segments
                    
                    if not self.worker.is_cancelled:
                        self.worker.safe_emit(
                            self.worker.log, 
                            f"✓ {target_lang} 번역 완료"
                        )
                except Exception as e:
                    if not self.worker.is_cancelled:
                        self.worker.safe_emit(
                            self.worker.log, 
                            f"❌ {target_lang} 번역 실패: {str(e)}"
                        )
                    results[target_lang] = None
                    
        return results
        
    def _translate_single_language(self, segments, target_lang, source_lang, 
                                  model, tokenizer, video_name):
        """단일 언어 번역 (스레드에서 실행)"""
        # 세그먼트 복사
        translated_segments = copy.deepcopy(segments)
        
        # 텍스트 추출
        texts = [seg['text'] for seg in translated_segments]
        total_texts = len(texts)
        
        # 배치 크기 결정 (30분 청크는 적당히)
        batch_size = 50 if total_texts > 100 else 30
        
        translated_texts = []
        
        # 배치 처리
        for i in range(0, total_texts, batch_size):
            if self.worker.is_cancelled:
                break
                
            batch = texts[i:i + batch_size]
            
            # 진행률 업데이트
            progress = int((i / total_texts) * 100)
            with self.lock:
                self.translation_progress[target_lang] = progress
                
            # 전체 진행률 계산
            avg_progress = sum(self.translation_progress.values()) / len(self.translation_progress)
            
            if not self.worker.is_cancelled:
                self.worker.safe_emit(
                    self.worker.progress,
                    video_name,
                    60 + int(avg_progress * 0.3),  # 60-90% 구간
                    f"번역 중... {target_lang}: {progress}%"
                )
            
            # 번역 실행
            try:
                # 소스 언어 설정
                tokenizer.src_lang = source_lang
                
                # 토크나이징
                model_inputs = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256  # 자막은 대부분 짧음
                )
                
                # 번역 생성
                generated_tokens = model.generate(
                    **model_inputs,
                    forced_bos_token_id=tokenizer.lang_code_to_id[target_lang],
                    max_length=256,
                    num_beams=2,  # 속도 우선
                    no_repeat_ngram_size=3,
                    use_cache=True
                )
                
                # 디코딩
                batch_translations = tokenizer.batch_decode(
                    generated_tokens, 
                    skip_special_tokens=True
                )
                
                translated_texts.extend(batch_translations)
                
            except Exception as e:
                # 오류 시 원문 유지
                translated_texts.extend(batch)
                if not self.worker.is_cancelled:
                    self.worker.safe_emit(
                        self.worker.log,
                        f"⚠️ {target_lang} 배치 {i//batch_size + 1} 번역 오류: {str(e)}"
                    )
        
        # 번역된 텍스트로 세그먼트 업데이트
        for seg, trans_text in zip(translated_segments, translated_texts):
            seg['text'] = trans_text
            
        return translated_segments
        
    def translate_for_chunk(self, segments: List[dict], 
                           target_languages: List[str],
                           source_lang: str,
                           chunk_index: int) -> Dict[str, List[dict]]:
        """청크용 번역 (워커에서 사용)"""
        
        if not target_languages:
            return {}
            
        # 진행률 표시용
        chunk_name = f"청크 {chunk_index}"
        
        # 병렬 번역 실행
        return self.translate_multiple_languages(
            segments, target_languages, source_lang, chunk_name
        )