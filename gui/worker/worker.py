"""
백그라운드 작업 처리를 위한 메인 Worker 스레드
"""
import threading
import traceback
import queue
import time
from PyQt6.QtCore import pyqtSignal
from auto_subtitle_llama.utils import filename, LANG_CODE_MAPPER

from .worker_base import BaseSubtitleWorker, ProgressParser
from .worker_process import VideoProcessor

class SubtitleWorker(BaseSubtitleWorker):
    """자막 생성 워커 스레드 - 메인 클래스"""
    
    # 병렬 처리용 시그널 추가
    workerProgress = pyqtSignal(int, int, int, str)  # worker_id, chunk_index, progress, status
    initWorkerProgress = pyqtSignal(int)  # num_workers
    
    def __init__(self, video_paths, settings):
        super().__init__(video_paths, settings)
        self.current_video_path = None
        self.progress_parser = ProgressParser(self)
        self.video_processor = VideoProcessor(self)

        self.last_progress_update = 0
        self.progress_throttle = 0.1  # 최소 0.1초 간격
        
        # 실시간 번역 설정 확인
        self.realtime_translation_enabled = settings.get('realtime_translation', True)
        self.translation_results = {}  # 실시간 번역 결과 저장
         
    def safe_emit_progress(self, filename, percent, status):
        """진행률 업데이트 throttling"""
        current_time = time.time()
        if current_time - self.last_progress_update >= self.progress_throttle:
            self.safe_emit(self.progress, filename, percent, status)
            self.last_progress_update = current_time
        
    def process_console_queue(self):
        """콘솔 큐 처리"""
        buffer = ""
        log_counter = 0
        log_skip_threshold = 10  # 10개마다 하나씩만 표시
        while not self.is_cancelled:
            try:
                text = self.console_queue.get(timeout=0.1)
                buffer += text
                
                # 줄바꿈이 있으면 로그로 전송
                if '\n' in buffer or '\r' in buffer:
                    lines = buffer.replace('\r\n', '\n').replace('\r', '\n').split('\n')
                    for line in lines[:-1]:
                        if line.strip() and not self.is_cancelled:
                            try:
                                # Whisper 다운로드 진행률 파싱
                                if self.progress_parser.parse_download_progress(line.strip()):
                                    continue
                                    
                                # Whisper 음성 인식 진행률 파싱
                                video_name = filename(self.current_video_path) if self.current_video_path else "처리 중"
                                if self.progress_parser.parse_transcribe_progress(line.strip(), video_name):
                                    continue
                                    
                                # 로그 스킵 (너무 많은 로그 방지)
                                log_counter += 1
                                if log_counter % log_skip_threshold == 0 or "error" in line.lower() or "warning" in line.lower():
                                    self.log.emit(line.strip())
                            except:
                                pass
                    buffer = lines[-1]
                    
            except queue.Empty:
                continue
            except Exception:
                break
                
    def run(self):
        """백그라운드 작업 실행"""
        try:
            # 콘솔 출력 캡처 시작
            self.capture_console_output()
            
            # 콘솔 큐 처리 스레드 시작
            self.console_thread = threading.Thread(target=self.process_console_queue)
            self.console_thread.daemon = True
            self.console_thread.start()
            
            # 모델 로드
            model = self.video_processor.load_model()
            if model is None or self.is_cancelled:
                return
            
            # 각 비디오 파일 처리
            for idx, video_path in enumerate(self.video_paths):
                if self.is_cancelled:
                    break
                    
                self.current_video_path = video_path
                
                # 실시간 번역이 활성화되고 번역 언어가 설정된 경우
                if (self.realtime_translation_enabled and 
                    self.settings.get('translate') and 
                    self.settings.get('languages')):
                    
                    # 현재 비디오의 언어 감지 (캐시에서 가져오거나 새로 감지)
                    detected_lang = self.video_processor.detect_language(video_path)
                    if detected_lang:
                        # mBART 소스 언어 코드
                        current_lang = LANG_CODE_MAPPER.get(detected_lang, [])
                        source_mbart_code = current_lang[1] if len(current_lang) > 1 else "en_XX"
                        
                        # 실시간 번역기 초기화
                        self.progress_parser.init_streaming_translator(
                            self.settings['languages'],
                            source_mbart_code
                        )
                        self.safe_emit(self.log, f"실시간 번역 모드로 처리 시작 ({detected_lang} → {', '.join(self.settings['languages'])})")
                
                # 비디오 처리
                self.video_processor.process_video(video_path, idx, len(self.video_paths))
                
                # 실시간 번역기 정지 및 통계
                if self.progress_parser.streaming_translator:
                    self.progress_parser.stop_streaming_translator()
                    self.progress_parser.streaming_translator = None
                    
            if not self.is_cancelled:
                self.safe_emit(self.log, f"✓ 모든 비디오 처리 완료")
                self.safe_emit(self.finished)
            
        except Exception as e:
            if not self.is_cancelled:
                self.safe_emit(self.error, str(e))
                traceback.print_exc()
          # UI에 에러 상태 전달

                if hasattr(self, 'current_video_path'):
                    video_name = filename(self.current_video_path)
                    self.safe_emit(self.fileCompleted, video_name, f"ERROR: {str(e)}")


        finally:
            # 정리 작업
            self.cleanup()
            self.video_processor.cleanup_temp_files()
            
            # 실시간 번역기 정리
            if self.progress_parser.streaming_translator:
                self.progress_parser.stop_streaming_translator()
            
    def cleanup(self):
        """정리 작업 - 부모 클래스 메서드 확장"""
        super().cleanup()
        # 추가 정리 작업이 필요한 경우 여기에 추가