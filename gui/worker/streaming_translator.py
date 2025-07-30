"""
실시간 스트리밍 번역을 위한 독립 모듈
재사용 가능한 형태로 설계
"""
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Callable, Optional
import time
from dataclasses import dataclass
from queue import Queue
import hashlib

@dataclass
class Segment:
    """자막 세그먼트 데이터 클래스"""
    start: float
    end: float
    text: str
    index: Optional[int] = None
    priority: int = 0

class StreamingTranslator:
    """
    실시간 스트리밍 번역 클래스
    - 독립적으로 작동
    - 콜백 기반으로 결과 전달
    - 다양한 환경에서 재사용 가능
    """
    
    def __init__(self, 
                 model=None, 
                 tokenizer=None,
                 target_languages: List[str] = None,
                 source_lang: str = "ko_KR",
                 callback: Callable = None,
                 buffer_size: int = 5,
                 max_workers: int = 3,
                 cache_enabled: bool = True):
        """
        Args:
            model: 번역 모델 (None이면 자동 로드)
            tokenizer: 토크나이저 (None이면 자동 로드)
            target_languages: 번역 대상 언어 리스트
            source_lang: 소스 언어
            callback: 번역 결과 콜백 함수
            buffer_size: 버퍼 크기 (배치 번역용)
            max_workers: 동시 번역 스레드 수
            cache_enabled: 캐시 사용 여부
        """
        self.model = model
        self.tokenizer = tokenizer
        self.target_languages = target_languages or []
        self.source_lang = source_lang
        self.callback = callback
        self.buffer_size = buffer_size
        self.max_workers = max_workers
        self.cache_enabled = cache_enabled
        
        # 모델이 없으면 로드
        if self.model is None or self.tokenizer is None:
            self._load_model()
            
        # 언어별 버퍼
        self.buffers = {lang: [] for lang in self.target_languages}
        self.buffer_lock = threading.Lock()
        
        # 번역 캐시
        self.translation_cache = {}
        self.cache_size = 1000
        self.max_queue_size = 1000  # 메모리 보호

        # 번역 결과 저장
        self.translation_results = {lang: [] for lang in self.target_languages}
        self.results_lock = threading.Lock()
        
        # 스레드 풀
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.is_running = True
        
        # 작업 추적
        self.pending_tasks = 0
        
        # 통계
        self.stats = {
            'total_segments': 0,
            'translated_segments': 0,
            'failed_segments': 0,
            'cache_hits': 0,
            'average_time': 0,
            'queue_size': 0
        }

    def _load_model(self):
        """번역 모델 로드"""
        try:
            from auto_subtitle_llama.utils import TranslatorManager
            manager = TranslatorManager()
            self.model, self.tokenizer = manager.get_translator()
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            raise

    def _get_cache_key(self, text: str, target_lang: str) -> str:
        """캐시 키 생성"""
        return f"{target_lang}:{hashlib.md5(text.encode()).hexdigest()[:16]}"

    def _check_cache(self, text: str, target_lang: str) -> Optional[str]:
        """캐시 확인"""
        if not self.cache_enabled:
            return None
        key = self._get_cache_key(text, target_lang)
        if key in self.translation_cache:
            self.stats['cache_hits'] += 1
            return self.translation_cache[key]
        return None

    def _save_to_cache(self, text: str, target_lang: str, translation: str):
        """캐시에 저장"""
        if not self.cache_enabled:
            return
        # 캐시 크기 제한
        if len(self.translation_cache) >= self.cache_size:
            # 가장 오래된 항목 제거 (간단한 FIFO)
            first_key = next(iter(self.translation_cache))
            del self.translation_cache[first_key]

        key = self._get_cache_key(text, target_lang)
        self.translation_cache[key] = translation

    def process_segment(self, segment: Dict, priority: int = 0):
        """
        새 세그먼트 처리

        Args:
            segment: {'start': float, 'end': float, 'text': str}
            priority: 우선순위 (높을수록 먼저 처리)
        """
        if not self.is_running:
            return

        # 빈 텍스트는 무시
        if not segment.get('text', '').strip():
            return

        # Segment 객체로 변환
        seg = Segment(
            start=segment['start'],
            end=segment['end'],
            text=segment['text'].strip(),
            priority=priority
        )
        # index가 있으면 추가

        if 'index' in segment:
            seg.index = segment['index']

        self.stats['total_segments'] += 1

        # 메모리 보호 - 큐가 너무 크면 강제 플러시
        with self.buffer_lock:
            total_buffered = sum(len(buffer) for buffer in self.buffers.values())
            if total_buffered > self.max_queue_size:
                self.flush_all()

        # 각 언어별로 처리
        for lang in self.target_languages:
            # 캐시 확인
            cached = self._check_cache(seg.text, lang)
            if cached:
                # 캐시된 결과 즉시 반환
                if self.callback:
                    segment_dict = {
                        'start': seg.start,
                        'end': seg.end,
                        'text': seg.text
                    }

                    self.callback({
                        'type': 'translation',
                        'segment': segment_dict,
                        'language': lang,
                        'translation': cached,
                        'from_cache': True,
                        'elapsed_time': 0
                    })
                continue

            # 버퍼에 추가
            with self.buffer_lock:
                self.buffers[lang].append(seg)

                # 즉시 번역 모드 또는 버퍼가 차면
                if self.buffer_size == 1 or len(self.buffers[lang]) >= self.buffer_size:
                    segments_to_translate = self.buffers[lang].copy()
                    self.buffers[lang].clear()

                    if len(segments_to_translate) == 1:
                        # 단일 번역
                        self.pending_tasks += 1
                        self.executor.submit(self._translate_single, segments_to_translate[0], lang)
                    else:
                        # 배치 번역
                        self.pending_tasks += 1
                        self.executor.submit(self._translate_batch, segments_to_translate, lang)

    def _translate_single(self, segment: Segment, target_lang: str):
        """단일 세그먼트 번역"""
        start_time = time.time()

        try:
            # 빠른 번역 설정
            self.tokenizer.src_lang = self.source_lang

            inputs = self.tokenizer(
                segment.text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            )

            outputs = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.lang_code_to_id[target_lang],
                max_length=256,  # 128 -> 256으로 증가
                num_beams=3,  # 1 -> 3으로 증가 (품질 향상)
                temperature=0.8,  # 추가 (다양성 조절)
                repetition_penalty=1.2,  # 추가 (반복 방지)
                do_sample=False
            )

            translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # 캐시에 저장
            self._save_to_cache(segment.text, target_lang, translation)


            # 결과 저장
            with self.results_lock:
                self.translation_results[target_lang].append({
                    'start': segment.start,
                    'end': segment.end,
                    'text': translation
                })

            # 통계 업데이트
            elapsed = time.time() - start_time
            self._update_stats(elapsed)

            # 콜백 호출
            if self.callback:
                # Segment 객체를 dict로 변환
                segment_dict = {
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment.text
                }

                self.callback({
                    'type': 'translation',
                    'segment': segment_dict,  # dataclass 대신 dict 전달
                    'language': target_lang,
                    'translation': translation,
                    'from_cache': False,
                    'elapsed_time': elapsed
                })

            self.stats['translated_segments'] += 1

        except Exception as e:
            self.stats['failed_segments'] += 1
            if self.callback:
                segment_dict = {
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment.text
                }

                self.callback({
                    'type': 'error',
                    'segment': segment_dict,
                    'language': target_lang,
                    'error': str(e)
                })
        finally:
            self.pending_tasks -= 1

    def _translate_batch(self, segments: List[Segment], target_lang: str):
        """배치 번역"""
        start_time = time.time()
        texts = [seg.text for seg in segments]

        try:
            # 배치 번역
            self.tokenizer.src_lang = self.source_lang

            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            )

            outputs = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.lang_code_to_id[target_lang],
                max_length=128,
                num_beams=1,
            )

            translations = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # 각 번역 결과 처리
            elapsed = time.time() - start_time
            avg_time = elapsed / len(segments)

            for seg, trans in zip(segments, translations):
                # 캐시에 저장
                self._save_to_cache(seg.text, target_lang, trans)

                # 결과 저장
                with self.results_lock:
                    self.translation_results[target_lang].append({
                        'start': seg.start,
                        'end': seg.end,
                        'text': trans
                    })

                if self.callback:
                    segment_dict = {
                        'start': seg.start,
                        'end': seg.end,
                        'text': seg.text
                    }

                    self.callback({
                        'type': 'translation',
                        'segment': segment_dict,
                        'language': target_lang,
                        'translation': trans,
                        'from_cache': False,
                        'elapsed_time': avg_time
                    })

            self.stats['translated_segments'] += len(segments)
            self._update_stats(elapsed, len(segments))

        except Exception as e:
            self.stats['failed_segments'] += len(segments)
            if self.callback:
                self.callback({
                    'type': 'batch_error',
                    'language': target_lang,
                    'error': str(e),
                    'segment_count': len(segments)
                })
        finally:
            self.pending_tasks -= 1

    def _update_stats(self, elapsed_time: float, count: int = 1):
        """통계 업데이트"""
        n = self.stats['translated_segments']
        avg = self.stats['average_time']
        self.stats['average_time'] = (avg * n + elapsed_time) / (n + count)

    def flush_all(self):
        """모든 버퍼 비우기"""
        with self.buffer_lock:
            for lang in self.target_languages:
                if self.buffers[lang]:
                    segments = self.buffers[lang].copy()
                    self.buffers[lang].clear()

                    if len(segments) == 1:
                        self.pending_tasks += 1
                        self.executor.submit(self._translate_single, segments[0], lang)
                    else:
                        self.pending_tasks += 1
                        self.executor.submit(self._translate_batch, segments, lang)

    def wait_for_completion(self, timeout: int = 30) -> int:
        """모든 번역 작업이 완료될 때까지 대기

        Returns:
            남은 작업 수
        """
        start_time = time.time()
        while self.pending_tasks > 0 and (time.time() - start_time) < timeout:
            time.sleep(0.1)

        # 강제로 executor 종료
        self.executor.shutdown(wait=True)

        return self.pending_tasks

    def stop(self):
        """번역기 정지"""
        self.is_running = False
        self.flush_all()
        self.executor.shutdown(wait=True)

    def get_stats(self) -> Dict:
        """통계 반환"""
        stats = self.stats.copy()
        stats['cache_size'] = len(self.translation_cache)
        return stats

    def get_results(self) -> Dict[str, List[Dict]]:
        """
        현재까지의 번역 결과 반환
        Returns:
            {language_code: [segments]}
        """
        with self.results_lock:
            # 각 언어별로 시간순 정렬
            sorted_results = {}
            for lang, segments in self.translation_results.items():
                sorted_results[lang] = sorted(segments, key=lambda x: x['start'])
            return sorted_results


# 사용 예시
def demo_callback(result):
    """콜백 함수 예시"""
    if result['type'] == 'translation':
        cache_str = " (cached)" if result.get('from_cache') else ""
        print(f"[{result['language']}] {result['translation']}{cache_str}")
    elif result['type'] == 'error':
        print(f"Error: {result['error']}")

# 독립적 사용 예시
if __name__ == "__main__":
    translator = StreamingTranslator(
        target_languages=["en_XX", "es_XX"],
        source_lang="ko_KR",
        callback=demo_callback,
        buffer_size=1  # 즉시 번역
    )
    
    # 테스트
    test_segments = [
        {'start': 0.0, 'end': 2.0, 'text': '안녕하세요'},
        {'start': 2.0, 'end': 4.0, 'text': '반갑습니다'},
        {'start': 4.0, 'end': 6.0, 'text': '오늘 날씨가 좋네요'}
    ]
    
    for seg in test_segments:
        translator.process_segment(seg)
        time.sleep(0.1)
    
    # 통계 출력
    print("\n통계:", translator.get_stats())
    
    translator.stop()