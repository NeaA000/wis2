"""
Worker 기본 클래스와 공통 유틸리티
"""
from PyQt6.QtCore import QThread, pyqtSignal
import sys
import queue
import threading
import os
import hashlib
import re
import time
import signal

class BaseSubtitleWorker(QThread):
    """Worker 기본 클래스 - 공통 기능 제공"""
    
    # 시그널 정의
    progress = pyqtSignal(str, int, str)  # 파일명, 퍼센트, 상태메시지
    fileCompleted = pyqtSignal(str, str)  # 파일명, 출력경로
    finished = pyqtSignal()
    error = pyqtSignal(str)
    log = pyqtSignal(str)
    # 실시간 번역 시그널 추가
    realtimeTranslation = pyqtSignal(dict, str, str)  # segment, language, translation
    
    def __init__(self, video_paths, settings):
        super().__init__()
        self.video_paths = video_paths
        self.settings = settings
        self.is_cancelled = False
        self.console_queue = queue.Queue()
        self.console_thread = None
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.cancel_event = threading.Event()
        self.active_threads = []
        
    def safe_emit(self, signal, *args):
        """안전하게 시그널 발생"""
        if not self.is_cancelled:
            try:
                signal.emit(*args)
            except:
                pass
                
    def get_cache_path(self, video_path, lang_code=None):
        """비디오별 캐시 파일 경로"""
        os.makedirs("cache", exist_ok=True)
        video_hash = hashlib.md5(video_path.encode()).hexdigest()[:8]
        
        if lang_code:
            return f"cache/{video_hash}_{lang_code}.pkl"
        else:
            return f"cache/{video_hash}_original.pkl"
            
    def cancel(self):
        """작업 취소"""
        self.is_cancelled = True
        self.cancel_event.set()
        self.safe_emit(self.log, "작업 취소 중...")
        
    # 콘솔 출력 관련 메서드들
    def capture_console_output(self):
        """콘솔 출력 캡처 시작"""
        class ConsoleCapture:
            def __init__(self, queue, original):
                self.queue = queue
                self.original = original
                
            def write(self, text):
                try:
                    self.queue.put(text)
                    self.original.write(text)
                except:
                    pass
                    
            def flush(self):
                try:
                    self.original.flush()
                except:
                    pass
        
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        sys.stdout = ConsoleCapture(self.console_queue, self.original_stdout)
        sys.stderr = ConsoleCapture(self.console_queue, self.original_stderr)
        
    def restore_console_output(self):
        """콘솔 출력 복원"""
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        
    def cleanup(self):
        """정리 작업"""
        try:
            self.restore_console_output()
            
            while not self.console_queue.empty():
                try:
                    self.console_queue.get_nowait()
                except:
                    break
                    
        except Exception as e:
            print(f"정리 중 오류: {e}")


class ProgressParser:
    """진행률 파싱 전용 클래스 - 실시간 번역 기능 추가"""
    
    def __init__(self, worker):
        self.worker = worker
        self.is_downloading_model = False
        self.is_transcribing = False
        self.transcribe_start_time = None
        self.audio_duration = None
        self.last_timestamp = 0
        self.total_segments = 0
        self.processed_segments = 0
        self.streaming_translator = None
        self.detected_segments = []  # 감지된 세그먼트 저장
        self.progress_tracker = None  # ProgressTracker 참조
        
    def init_streaming_translator(self, target_languages, source_lang):
        """스트리밍 번역기 초기화"""
        if not target_languages:
            return
            
        from gui.worker.streaming_translator import StreamingTranslator
        
        def translation_callback(result):
            """번역 결과 콜백"""
            if result['type'] == 'translation':

                # segment 타입 확인 및 안전한 dict 변환
                segment = result['segment']

                if isinstance(segment, dict):
                    segment_dict = segment
                elif hasattr(segment, '__dict__'):
                    segment_dict = segment.__dict__
                else:
                    segment_dict = {'start': 0, 'end': 0, 'text': str(segment)}

                self.worker.safe_emit(
                    self.worker.realtimeTranslation,
                    segment_dict,  # 안전하게 변환된 dict
                    result['language'],
                    result['translation']
                )
                # 로그에도 표시
                if not result.get('from_cache') and self.worker.settings.get('realtime_log', True):
                    self.worker.safe_emit(
                        self.worker.log,
                        f"[실시간 번역] {result['language']}: {result['translation'][:50]}..."
                    )
            elif result['type'] == 'error':
                self.worker.safe_emit(
                    self.worker.log,
                    f"[번역 오류] {result['language']}: {result['error']}"
                )
        
        self.streaming_translator = StreamingTranslator(
            target_languages=target_languages,
            source_lang=source_lang,
            callback=translation_callback,
            buffer_size=1,  # 실시간 모드
            max_workers=len(target_languages),  # 언어별 스레드
            cache_enabled=True
        )
        
        self.worker.safe_emit(self.worker.log, f"실시간 번역 활성화: {', '.join(target_languages)}")
        
    def stop_streaming_translator(self):
        """스트리밍 번역기 정지"""
        if self.streaming_translator:
            self.streaming_translator.stop()
            stats = self.streaming_translator.get_stats()
            self.worker.safe_emit(
                self.worker.log,
                f"실시간 번역 통계 - 처리: {stats['translated_segments']}, "
                f"캐시 히트: {stats['cache_hits']}, 평균 시간: {stats['average_time']:.2f}초"
            )
        
    def parse_download_progress(self, line):
        """Whisper 모델 다운로드 진행률 파싱"""
        # 다운로드 진행률 패턴 매칭
        progress_pattern = r'(\d+)%\|.*?\|\s*(\d+\.?\d*[KMG]?)\/(\d+\.?\d*[KMG]?)\s*\[([^\]]+)\]'
        match = re.search(progress_pattern, line)
        
        if match:
            percent = int(match.group(1))
            downloaded = match.group(2)
            total = match.group(3)
            time_info = match.group(4)
            
            # 속도 정보 추출
            speed_match = re.search(r'(\d+\.?\d*[KMG]?B/s)', time_info)
            speed = speed_match.group(1) if speed_match else ""
            
            # 남은 시간 추출
            remaining_match = re.search(r'<(\d+:\d+)', time_info)
            remaining = remaining_match.group(1) if remaining_match else ""
            
            if not self.is_downloading_model:
                self.is_downloading_model = True
                self.worker.safe_emit(self.worker.log, f"Whisper 모델 다운로드 시작... (총 {total})")
            
            # 상태 메시지 구성
            status = f"다운로드 중... {downloaded}/{total}"
            if speed:
                status += f" ({speed})"
            if remaining:
                status += f" - 남은 시간: {remaining}"
                
            self.worker.safe_emit(self.worker.progress, "모델 다운로드", percent, status)
            
            if percent >= 100:
                self.is_downloading_model = False
                self.worker.safe_emit(self.worker.log, "✓ 모델 다운로드 완료")
                
            return True
            
        # 다운로드 시작 메시지 감지
        if "Downloading" in line and ".pt" in line:
            self.is_downloading_model = True
            model_match = re.search(r'Downloading\s+(\S+\.pt)', line)
            if model_match:
                model_file = model_match.group(1)
                self.worker.safe_emit(self.worker.log, f"Whisper 모델 다운로드 준비 중... ({model_file})")
            else:
                self.worker.safe_emit(self.worker.log, "Whisper 모델 다운로드 준비 중...")
            return True
            
        # 다운로드 오류 감지
        if self.is_downloading_model and ("error" in line.lower() or "failed" in line.lower()):
            self.is_downloading_model = False
            self.worker.safe_emit(self.worker.log, f"❌ 모델 다운로드 오류: {line}")
            return True
            
        return False
    
    def set_audio_duration(self, duration):
        """오디오 길이 설정 (초 단위)"""
        self.audio_duration = duration
        self.last_timestamp = 0
        self.total_segments = 0
        self.processed_segments = 0
    
    def parse_transcribe_progress(self, line, video_name):
        """Whisper 음성 인식 진행률 파싱 - 실시간 번역 연동"""
        if not self.is_transcribing:
            return False
            
        # 여러 가지 Whisper 출력 패턴 매칭
        # 패턴 1: [00:00.000 --> 00:02.000]  텍스트
        timestamp_pattern = r'\[(\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}\.\d{3})\]\s*(.+)?'
        match = re.search(timestamp_pattern, line)

        if not match:
            # 패턴 2: 00:00.000 --> 00:02.000  텍스트
            timestamp_pattern2 = r'(\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}\.\d{3})\s*(.+)?'
            match = re.search(timestamp_pattern2, line)
        
        if match:
            start_time_str = match.group(1)
            current_time_str = match.group(2)
            text = match.group(3) if len(match.groups()) >= 3 else ""
            
            # 시간을 초 단위로 변환
            start_seconds = self._time_to_seconds(start_time_str)
            end_seconds = self._time_to_seconds(current_time_str)
            
            self.last_timestamp = end_seconds
            self.processed_segments += 1
            
            # 실시간 번역을 위한 세그먼트 생성
            if text and text.strip() and self.streaming_translator:
                segment = {
                    'start': start_seconds,
                    'end': end_seconds,
                    'text': text.strip()
                }
                # 감지된 세그먼트 저장
                self.detected_segments.append(segment)
                # 스트리밍 번역기로 전송
                self.streaming_translator.process_segment(segment)
            
            # 자막 텍스트를 로그에도 표시
            if text and text.strip():
                self.worker.safe_emit(self.worker.log, f"{match.group(0)}")
            
            # ProgressTracker 업데이트
            if self.progress_tracker and self.audio_duration:
                self.progress_tracker.update_transcribe_progress(
                    end_seconds, self.audio_duration, text
                )
                return True  # 진행률 업데이트 완료
                
            # 실제 오디오 길이를 기반으로 진행률 계산
            if self.audio_duration and self.audio_duration > 0:
                percent = min(int((end_seconds / self.audio_duration) * 100), 95)
                
                # 남은 시간 계산
                remaining_seconds = self.audio_duration - end_seconds
                if remaining_seconds > 0:
                    elapsed = time.time() - self.transcribe_start_time if self.transcribe_start_time else 0
                    if elapsed > 0 and end_seconds > 0:
                        # 처리 속도 기반 예측
                        speed = end_seconds / elapsed
                        estimated_remaining = remaining_seconds / speed if speed > 0 else remaining_seconds
                        
                        if estimated_remaining > 60:
                            remaining_min = int(estimated_remaining // 60)
                            remaining_sec = int(estimated_remaining % 60)
                            status = f"음성 인식 중... [{current_time_str}/{self._format_time(self.audio_duration)}] - 예상 남은 시간: {remaining_min}분 {remaining_sec}초"
                        else:
                            status = f"음성 인식 중... [{current_time_str}/{self._format_time(self.audio_duration)}] - 예상 남은 시간: {int(estimated_remaining)}초"
                    else:
                        status = f"음성 인식 중... [{current_time_str}/{self._format_time(self.audio_duration)}]"
                else:
                    status = f"음성 인식 중... [{current_time_str}/{self._format_time(self.audio_duration)}]"
            else:
                # 오디오 길이를 모르는 경우
                estimated_total = max(600, end_seconds * 1.2)  # 최소 10분
                percent = min(int((end_seconds / estimated_total) * 100), 95)
                status = f"음성 인식 중... [{current_time_str}]"

            # 텍스트 미리보기 추가 (너무 길면 자르기)
            if text and text.strip():
                # 진행률 표시에는 짧게만
                preview = text.strip()[:30] + "..." if len(text.strip()) > 30 else text.strip()
                status += f" - {preview}"
            
            self.worker.safe_emit(self.worker.progress, video_name, 40 + int(percent * 0.5), status)
            
            return True
            
        # 언어 감지 메시지   
        if "Detecting language" in line:
            self.worker.safe_emit(self.worker.progress, video_name, 35, "언어 감지 중...")
            return True
            
        # 언어 감지 결과
        if "Detected language:" in line:
            lang_match = re.search(r'Detected language:\s*(\w+)', line)
            if lang_match:
                detected_lang = lang_match.group(1)
                self.worker.safe_emit(self.worker.progress, video_name, 38, f"감지된 언어: {detected_lang}")
            return True
            
        # 완료 메시지
        if "Transcription complete" in line or "Done" in line or "transcription finished" in line.lower():
            # 마지막 버퍼 비우기
            if self.streaming_translator:
                self.streaming_translator.flush_all()
                
            self.worker.safe_emit(self.worker.progress, video_name, 90, "음성 인식 완료")
            self.is_transcribing = False
            return True
            
        # 처리 메시지들
        if "Processing" in line or "Performing" in line:
            # 특별한 처리 단계 감지
            if "silence" in line.lower():
                self.worker.safe_emit(self.worker.progress, video_name, 42, "무음 구간 처리 중...")
            elif "speech" in line.lower():
                self.worker.safe_emit(self.worker.progress, video_name, 45, "음성 구간 분석 중...")
            else:
                self.worker.safe_emit(self.worker.log, f"처리 중: {line.strip()}")
            return True
            
        # 프로그레스 바 패턴 (일부 Whisper 버전)
        progress_bar_pattern = r'(\d+)%\|'
        bar_match = re.search(progress_bar_pattern, line)
        if bar_match:
            percent = int(bar_match.group(1))
            status = f"음성 인식 처리 중... {percent}%"
            self.worker.safe_emit(self.worker.progress, video_name, 40 + int(percent * 0.5), status)
            return True
            
        return False
        
    def _time_to_seconds(self, time_str):
        """MM:SS.mmm 형식을 초로 변환"""
        parts = time_str.split(':')
        minutes = int(parts[0])
        seconds = float(parts[1])
        return minutes * 60 + seconds
        
    def _format_time(self, seconds):
        """초를 MM:SS.mmm 형식으로 변환"""
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes:02d}:{secs:06.3f}"
        
    def get_detected_segments(self):
        """감지된 세그먼트 반환"""
        return self.detected_segments.copy()