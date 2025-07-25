"""
병렬 비디오 처리를 위한 모듈
긴 영상을 청크로 분할하여 병렬 처리하고 결과를 병합
"""
import os
import tempfile
import multiprocessing as mp
from multiprocessing import Queue, Process
import whisper
import ffmpeg
import pickle
from typing import List, Dict, Tuple, Optional
import time
import traceback
from dataclasses import dataclass
import threading
import re

# utils 함수들 import (수정됨)
from auto_subtitle_llama.utils import (
    write_srt, format_timestamp, adjust_timestamps, 
    merge_overlapping_segments, find_best_split_point
)


@dataclass
class ChunkInfo:
    """비디오 청크 정보"""
    index: int
    start_time: float
    end_time: float
    duration: float
    temp_path: str
    overlap_start: Optional[float] = None
    overlap_end: Optional[float] = None


class ParallelVideoProcessor:
    """병렬 비디오 처리 클래스"""
    
    def __init__(self, 
                 model_name: str = "turbo",
                 num_workers: int = None,
                 chunk_duration: int = 1800,  # 30분
                 overlap_duration: int = 60,   # 1분
                 verbose: bool = True):
        """
        Args:
            model_name: Whisper 모델 이름
            num_workers: 워커 프로세스 수 (None이면 CPU 코어 수)
            chunk_duration: 청크 길이 (초)
            overlap_duration: 오버랩 길이 (초)
            verbose: 상세 로그 출력 여부
        """
        self.model_name = model_name
        self.num_workers = num_workers or mp.cpu_count()
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration
        self.verbose = verbose
        
    def get_video_duration(self, video_path: str) -> float:
        """비디오 길이 구하기"""
        try:
            probe = ffmpeg.probe(video_path)
            video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            duration = float(probe['format']['duration'])
            return duration
        except Exception as e:
            print(f"비디오 길이 확인 실패: {e}")
            return 0
            
    def split_video_with_overlap(self, video_path: str, audio_path: str) -> List[ChunkInfo]:
        """비디오를 오버랩이 있는 청크로 분할"""
        duration = self.get_video_duration(video_path)
        chunks = []
        
        # 임시 디렉토리 생성
        temp_dir = tempfile.mkdtemp(prefix="whisper_chunks_")
        
        current_time = 0
        chunk_index = 0
        
        while current_time < duration:
            # 청크 시작/종료 시간 계산
            start_time = current_time
            end_time = min(current_time + self.chunk_duration, duration)
            
            # 마지막 청크가 아니면 오버랩 추가
            if end_time < duration:
                overlap_end = min(end_time + self.overlap_duration, duration)
            else:
                overlap_end = end_time
                
            # 청크 정보 생성
            chunk_path = os.path.join(temp_dir, f"chunk_{chunk_index:03d}.wav")
            
            chunk_info = ChunkInfo(
                index=chunk_index,
                start_time=start_time,
                end_time=end_time,
                duration=overlap_end - start_time,
                temp_path=chunk_path,
                overlap_start=end_time if end_time < duration else None,
                overlap_end=overlap_end if end_time < duration else None
            )
            
            # FFmpeg로 오디오 청크 추출
            self._extract_audio_chunk(audio_path, chunk_info)
            
            chunks.append(chunk_info)
            
            # 다음 청크로 이동
            current_time = end_time
            chunk_index += 1
            
        return chunks
        
    def _extract_audio_chunk(self, audio_path: str, chunk_info: ChunkInfo):
        """오디오 청크 추출"""
        try:
            ffmpeg.input(
                audio_path,
                ss=chunk_info.start_time,
                t=chunk_info.duration
            ).output(
                chunk_info.temp_path,
                acodec='pcm_s16le',
                ac=1,
                ar='16k'
            ).run(quiet=True, overwrite_output=True)
            
            if self.verbose:
                print(f"청크 {chunk_info.index} 추출: "
                      f"{format_timestamp(chunk_info.start_time)} ~ "
                      f"{format_timestamp(chunk_info.start_time + chunk_info.duration)}")
                
        except Exception as e:
            print(f"청크 추출 실패: {e}")
            raise
            
    def process_chunk_worker(self, chunk_queue: Queue, result_queue: Queue, 
                           task_args: dict, worker_id: int):
        """워커 프로세스에서 청크 처리"""
        # 각 워커에서 모델 로드
        print(f"[Worker {worker_id}] Whisper 모델 로드 중...")
        model = whisper.load_model(self.model_name)
        print(f"[Worker {worker_id}] 모델 로드 완료")
        
        while True:
            try:
                 # 취소 확인 추가
                if hasattr(threading.current_thread(), 'cancel_event'):
                    if threading.current_thread().cancel_event.is_set():
                        break
                        
                chunk_info = chunk_queue.get(timeout=1)

                if chunk_info is None:  # 종료 신호
                    break
                    
                print(f"[Worker {worker_id}] 청크 {chunk_info.index} 처리 시작")
                
                # 진행률 추정을 위한 시작 시간
                start_time = time.time()
                
                # 청크 길이에 따른 예상 처리 시간 (대략적)
                # 일반적으로 Whisper는 실시간의 5-10배 속도로 처리
                estimated_duration = chunk_info.duration / 1.1  # 5배속 가정
                
                # 주기적으로 진행률 업데이트
                def update_progress():
                    elapsed = time.time() - start_time
                    progress = min(int((elapsed / estimated_duration) * 95), 95)  # 최대 95%
                    
                    result_queue.put({
                        'type': 'worker_progress',
                        'worker_id': worker_id,
                        'chunk_index': chunk_info.index,
                        'progress': progress,
                        'status': f"처리 중... ({elapsed:.1f}초)"
                    })
                
                # 백그라운드 스레드로 진행률 업데이트
                progress_stop = threading.Event()
                
                def progress_updater():
                    while not progress_stop.is_set():
                        update_progress()
                        time.sleep(2)  # 2초마다 업데이트
                
                progress_thread = threading.Thread(target=progress_updater)
                progress_thread.daemon = True
                progress_thread.start()
                
                # Whisper로 음성 인식 (verbose=False로 변경)
                result = model.transcribe(
                    chunk_info.temp_path,
                    task=task_args.get('task', 'transcribe'),
                    language=task_args.get('language'),
                    verbose=False,  # False로 변경
                    fp16=False  # CPU에서는 FP32 사용
                )
                
                # 진행률 업데이트 중지
                progress_stop.set()
                progress_thread.join(timeout=1)
                
                # 타임스탬프 조정 (utils 함수 사용)
                adjusted_segments = adjust_timestamps(
                    result['segments'], 
                    chunk_info.start_time
                )
                
                # 완료 시 100% 전송
                result_queue.put({
                    'type': 'worker_progress',
                    'worker_id': worker_id,
                    'chunk_index': chunk_info.index,
                    'progress': 100,
                    'status': '완료'
                })
                
                # 결과 전송
                result_queue.put({
                    'type': 'result',
                    'chunk_info': chunk_info,
                    'segments': adjusted_segments,
                    'language': result.get('language', task_args.get('language'))
                })
                
                print(f"[Worker {worker_id}] 청크 {chunk_info.index} 완료")
                
            except Exception as e:
                print(f"[Worker {worker_id}] 오류: {e}")
                traceback.print_exc()
                result_queue.put({
                    'type': 'error',
                    'chunk_info': chunk_info if 'chunk_info' in locals() else None,
                    'error': str(e)
                })
        
    def process_video_parallel(self, video_path: str, audio_path: str, 
                             task_args: dict, progress_callback=None) -> dict:
        """비디오를 병렬로 처리"""
        start_time = time.time()
        
        # 1. 비디오를 청크로 분할
        if progress_callback:
            progress_callback("비디오 분할 중...", 10)
            
        chunks = self.split_video_with_overlap(video_path, audio_path)
        print(f"총 {len(chunks)}개 청크로 분할됨")
        
        # 2. 병렬 처리를 위한 큐 설정
        chunk_queue = mp.Queue()
        result_queue = mp.Queue()
        
        # 청크를 큐에 추가
        for chunk in chunks:
            chunk_queue.put(chunk)
            
        # 종료 신호 추가
        for _ in range(self.num_workers):
            chunk_queue.put(None)
            
        # 3. 워커 프로세스 시작
        if progress_callback:
            progress_callback("병렬 처리 시작...", 20)
            
        workers = []
        for i in range(self.num_workers):
            worker = Process(
                target=self.process_chunk_worker,
                args=(chunk_queue, result_queue, task_args, i)
            )
            worker.start()
            workers.append(worker)
            
        # 4. 결과 수집
        results = []
        errors = []
        processed = 0
        
        while processed < len(chunks):
            result = result_queue.get()
            
            # 타입별로 처리
            if result.get('type') == 'worker_progress':
                # 워커 진행률은 progress_callback으로 전달
                if progress_callback:
                    progress_callback(result, 0)  # 진행률 데이터 전달
            elif result.get('type') == 'error':
                errors.append(result)
                processed += 1
            elif result.get('type') == 'result':
                results.append(result)
                processed += 1
            
            if progress_callback and processed > 0:
                progress = 20 + (processed / len(chunks) * 60)
                progress_callback(
                    f"청크 처리 중... ({processed}/{len(chunks)})",
                    int(progress)
                )
                
        # 5. 워커 종료 대기
        for worker in workers:
            worker.join()
            
        # 6. 결과 병합
        if progress_callback:
            progress_callback("결과 병합 중...", 85)
            
        # 청크 인덱스로 정렬
        results.sort(key=lambda x: x['chunk_info'].index)
        
        # 오버랩 부분 병합
        merged_segments = self.merge_overlapping_results(results)
        
        # 7. 임시 파일 정리
        if progress_callback:
            progress_callback("임시 파일 정리 중...", 95)
            
        self._cleanup_temp_files(chunks)
        
        elapsed_time = time.time() - start_time
        print(f"병렬 처리 완료: {elapsed_time:.1f}초")
        
        return {
            'segments': merged_segments,
            'language': results[0]['language'] if results else 'unknown',
            'processing_time': elapsed_time,
            'num_chunks': len(chunks),
            'errors': errors
        }
        
    def merge_overlapping_results(self, results: List[dict]) -> List[dict]:
        """오버랩된 결과 병합"""
        if not results:
            return []
            
        all_segments = []
        
        for i, result in enumerate(results):
            segments = result['segments']
            chunk_info = result['chunk_info']
            
            if i == 0:
                # 첫 번째 청크는 모두 추가
                all_segments.extend(segments)
            else:
                # 오버랩 구간에서 최적의 병합 지점 찾기
                if chunk_info.overlap_start is not None:
                    merge_point = self._find_best_merge_point(
                        all_segments,
                        segments,
                        chunk_info.overlap_start
                    )
                    
                    # 병합 지점 이후의 세그먼트만 추가
                    for segment in segments:
                        if segment['start'] >= merge_point:
                            all_segments.append(segment)
                else:
                    # 오버랩이 없으면 모두 추가
                    all_segments.extend(segments)
        
        # utils의 merge_overlapping_segments 사용하여 추가 정리
        return merge_overlapping_segments(all_segments)
        
    def _find_best_merge_point(self, prev_segments: List[dict], 
                              curr_segments: List[dict], 
                              overlap_start: float) -> float:
        """오버랩 구간에서 최적의 병합 지점 찾기"""
        # 오버랩 구간의 텍스트 비교
        prev_overlap_texts = []
        curr_overlap_texts = []
        
        # 이전 청크의 오버랩 구간 텍스트
        for seg in prev_segments:
            if seg['start'] >= overlap_start:
                prev_overlap_texts.append((seg['start'], seg['text'].strip()))
                
        # 현재 청크의 오버랩 구간 텍스트
        for seg in curr_segments:
            if seg['start'] >= overlap_start and seg['start'] < overlap_start + self.overlap_duration:
                curr_overlap_texts.append((seg['start'], seg['text'].strip()))
                
        # 동일한 텍스트 찾기
        for curr_time, curr_text in curr_overlap_texts:
            for prev_time, prev_text in prev_overlap_texts:
                if curr_text == prev_text and len(curr_text) > 10:  # 충분히 긴 텍스트
                    # 동일한 텍스트를 찾았으면 그 지점에서 병합
                    return curr_time
                
                # 부분적으로 겹치는 텍스트 찾기
                if len(curr_text) > 20 and len(prev_text) > 20:
                    # 텍스트의 끝부분과 시작부분이 겹치는지 확인
                    overlap_length = min(len(prev_text), len(curr_text), 30)
                    if prev_text[-overlap_length:] in curr_text[:overlap_length*2]:
                        # utils의 find_best_split_point 활용
                        split_point = find_best_split_point(prev_text)
                        if split_point < len(prev_text):
                            return curr_time
                    
        # 동일한 텍스트를 못 찾으면 오버랩 시작 지점에서 병합
        return overlap_start
        
    def _cleanup_temp_files(self, chunks: List[ChunkInfo]):
        """임시 파일 정리"""
        for chunk in chunks:
            try:
                if os.path.exists(chunk.temp_path):
                    os.remove(chunk.temp_path)
            except Exception as e:
                print(f"임시 파일 삭제 실패: {e}")
                
        # 임시 디렉토리 삭제
        if chunks:
            temp_dir = os.path.dirname(chunks[0].temp_path)
            try:
                os.rmdir(temp_dir)
            except Exception as e:
                print(f"임시 디렉토리 삭제 실패: {e}")


def process_long_video_parallel(video_path: str, audio_path: str, 
                              model_name: str, task_args: dict,
                              num_workers: int = None,
                              chunk_duration: int = 1800,
                              progress_callback=None) -> dict:
    """긴 비디오를 병렬로 처리하는 헬퍼 함수"""
    processor = ParallelVideoProcessor(
        model_name=model_name,
        num_workers=num_workers,
        chunk_duration=chunk_duration
    )
    
    return processor.process_video_parallel(
        video_path=video_path,
        audio_path=audio_path,
        task_args=task_args,
        progress_callback=progress_callback
    )