"""
비디오 처리 관련 로직
"""
from auto_subtitle_llama.cli import get_audio
from auto_subtitle_llama.utils import filename, write_srt, get_text_batch, replace_text_batch, LANG_CODE_MAPPER, remove_duplicate_segments, advanced_remove_duplicates
import whisper
import ffmpeg
import pickle
import os
import copy
import time
import tempfile
import traceback
import threading

class VideoProcessor:
    """비디오 처리 전담 클래스"""
    
    def __init__(self, worker):
        self.worker = worker
        self.model = None
        
    def load_model(self):
        """Whisper 모델 로드"""
        if not self.worker.is_cancelled:
            self.worker.safe_emit(self.worker.log, f"Whisper {self.worker.settings['model']} 모델 로드 중...")
            
            if len(self.worker.video_paths) > 0:
                video_name = filename(self.worker.video_paths[0])
                self.worker.safe_emit(self.worker.progress, video_name, 5, "모델 준비 중...")
                
            self.model = whisper.load_model(self.worker.settings['model'])
            self.worker.safe_emit(self.worker.log, f"✓ 모델 로드 완료")
            
        return self.model
        
    def process_video(self, video_path, idx, total):
        """단일 비디오 처리"""
        video_name = filename(video_path)
        self.worker.safe_emit(self.worker.log, f"\n[{idx+1}/{total}] {video_name} 처리 시작")
        
        # 현재 비디오 경로 저장 (병렬 처리용)
        self.worker.current_video_path = video_path
        
        try:
            # 1. 캐시 확인
            cache_path = self.worker.get_cache_path(video_path)
            if os.path.exists(cache_path) and not self.worker.settings.get('force_reprocess', False):
                self.worker.safe_emit(self.worker.progress, video_name, 10, "캐시에서 불러오는 중...")
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    
                if not self.worker.is_cancelled:
                    self.process_cached_video(video_path, cached_data)
                return
            
            if self.worker.is_cancelled:
                return
                
            # 2. 오디오 추출
            self.worker.safe_emit(self.worker.progress, video_name, 20, "오디오 추출 중...")
            self.worker.safe_emit(self.worker.log, "오디오 스트림 추출 중...")
            audio_paths = get_audio([video_path])
            self.worker.safe_emit(self.worker.log, "✓ 오디오 추출 완료")
            
            if self.worker.is_cancelled:
                return
                
            # 3. 음성 인식
            result, detected_lang = self.transcribe_audio(
                audio_paths[video_path], 
                video_name
            )
            
            if result is None or self.worker.is_cancelled:
                return
                
            # 중복 세그먼트 제거
            original_count = len(result["segments"])
            if self.worker.settings.get('advanced_duplicate_removal', True):
                # 고급 중복 제거 사용
                result["segments"] = advanced_remove_duplicates(result["segments"])
            else:
                # 기본 중복 제거 사용
                result["segments"] = remove_duplicate_segments(result["segments"])
            if original_count != len(result["segments"]):
                self.worker.safe_emit(self.worker.log, f"중복 자막 {original_count - len(result['segments'])}개 제거됨")
            
            # 4. 자막 저장
            srt_path = self.save_subtitles(video_name, result["segments"])
            subtitles = {video_path: srt_path}
            
            # 5. 캐시 저장
            cache_data = {
                'subtitles': subtitles,
                'detected_lang': detected_lang,
                'audio_paths': audio_paths,
                'segments': result["segments"]
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            
            self.worker.safe_emit(self.worker.log, f"✓ 원본 자막 생성 완료: {srt_path}")
            
            if self.worker.is_cancelled:
                return
                
            # 6. 번역 처리
            if self.worker.settings['translate'] and self.worker.settings['languages']:
                self.process_translations(
                    video_path, audio_paths, detected_lang, result["segments"]
                )
            
            if self.worker.is_cancelled:
                return
                
            # 7. 비디오 임베딩
            if not self.worker.settings['srt_only']:
                self.embed_subtitles(video_path, subtitles, detected_lang)
                
            # 완료
            self.worker.safe_emit(self.worker.fileCompleted, video_name, srt_path)
            self.worker.safe_emit(self.worker.progress, video_name, 100, "완료!")
            
        except Exception as e:
            if not self.worker.is_cancelled:
                self.worker.safe_emit(self.worker.log, f"❌ {video_name} 처리 중 오류: {str(e)}")
                traceback.print_exc()
                
    def transcribe_audio(self, audio_path, video_name):
        """음성 인식 수행"""
        # 언어 감지
        self.worker.safe_emit(self.worker.progress, video_name, 30, "언어 감지 중...")
        
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio, self.model.dims.n_mels).to(self.model.device)
        _, probs = self.model.detect_language(mel)
        detected_lang = max(probs, key=probs.get)
        self.worker.safe_emit(self.worker.log, f"감지된 언어: {detected_lang}")

        if self.worker.is_cancelled:
            return None, None
        
        # 오디오 길이 확인
        try:
            probe = ffmpeg.probe(audio_path)
            duration = float(probe['streams'][0]['duration'])
            self.worker.safe_emit(self.worker.log, f"오디오 길이: {duration//60:.0f}분 {duration%60:.0f}초")
        except:
            duration = 0
            self.worker.safe_emit(self.worker.log, "오디오 길이 확인 실패, 단일 처리 모드 사용")
        
        # 30분(1800초) 이상이고 병렬 처리 설정이 켜져있으면 병렬 처리
        if duration > 1800 and self.worker.settings.get('parallel_processing', True):
            self.worker.safe_emit(
                self.worker.log, 
                f"긴 오디오 감지 ({duration//60:.0f}분). 병렬 처리 시작..."
            )
            
            # 병렬 처리
            try:
                from gui.parallel_processor import process_long_video_parallel
            except ImportError:
                self.worker.safe_emit(self.worker.log, "❌ 병렬 처리 모듈 로드 실패, 단일 처리로 전환")
                return self._transcribe_single(audio_path, video_name, detected_lang)
            
            # 진행률 콜백 함수
            def progress_callback(message, percent):
                self.worker.safe_emit(
                    self.worker.progress, 
                    video_name, 
                    40 + int(percent * 0.4),  # 40~80% 구간 사용
                    message
                )
            
            # video_path 가져오기
            video_path = self.worker.current_video_path
            
            result = process_long_video_parallel(
                video_path=video_path,
                audio_path=audio_path,
                model_name=self.worker.settings['model'],
                task_args={
                    'task': 'transcribe',
                    'language': detected_lang
                },
                num_workers=self.worker.settings.get('num_workers', 3),
                progress_callback=progress_callback
            )
            
            # 에러 체크
            if result.get('errors'):
                self.worker.safe_emit(
                    self.worker.log, 
                    f"⚠️ {len(result['errors'])}개 청크에서 오류 발생"
                )
            
            # 병렬 처리 결과 포맷 맞추기
            formatted_result = {
                'segments': result['segments'],
                'language': detected_lang
            }
            
            self.worker.safe_emit(
                self.worker.log, 
                f"✓ 병렬 처리 완료: {len(result['segments'])}개 세그먼트"
            )
            
            return formatted_result, detected_lang
            
        else:
            # 30분 미만 또는 병렬 처리 비활성화 시 단일 처리
            return self._transcribe_single(audio_path, video_name, detected_lang)
    
    def _transcribe_single(self, audio_path, video_name, detected_lang):
        """단일 스레드 음성 인식"""
        self.worker.safe_emit(self.worker.progress, video_name, 40, "음성 인식 시작...")
        
        # 진행률 파서의 플래그 설정
        self.worker.progress_parser.is_transcribing = True
        self.worker.progress_parser.transcribe_start_time = time.time()
        
        result = None
        transcribe_thread = None
        try:
            # 취소 가능한 transcribe 구현
            def transcribe_task():
                nonlocal result
                try:
                    result = self.model.transcribe(
                        audio_path,
                        task="transcribe",
                        language=detected_lang,
                        verbose=True,
                        fp16=False
                    )
                except Exception as e:
                    if not self.worker.is_cancelled:
                        self.worker.safe_emit(self.worker.log, f"❌ 음성 인식 오류: {str(e)}")
            
            transcribe_thread = threading.Thread(target=transcribe_task)
            self.worker.active_threads.append(transcribe_thread)
            transcribe_thread.start()

            # 주기적으로 취소 확인
            while transcribe_thread.is_alive() and not self.worker.is_cancelled:
                transcribe_thread.join(timeout=0.5)
                
            if self.worker.is_cancelled and transcribe_thread.is_alive():
                # 강제 종료는 하지 않고 완료 대기 (최대 5초)
                transcribe_thread.join(timeout=5.0)
            
        finally:
            self.worker.progress_parser.is_transcribing = False
            if transcribe_thread and transcribe_thread in self.worker.active_threads:  
                self.worker.active_threads.remove(transcribe_thread)
            
        return result, detected_lang
        
    def save_subtitles(self, video_name, segments):
        """자막 파일 저장"""
        srt_path = os.path.join(self.worker.settings['output_dir'], f"{video_name}.srt")
        with open(srt_path, "w", encoding="utf-8") as srt_file:
            write_srt(segments, srt_file)
        return srt_path
        
    def process_cached_video(self, video_path, cached_data):
        """캐시된 비디오 처리"""
        if self.worker.is_cancelled:
            return
            
        video_name = filename(video_path)
        self.worker.safe_emit(self.worker.progress, video_name, 50, "캐시 데이터 로드 완료")
        
        # 번역 처리
        if self.worker.settings['translate'] and self.worker.settings['languages'] and not self.worker.is_cancelled:
            segments = cached_data.get('segments', [])
            self.process_translations(
                video_path, 
                cached_data['audio_paths'],
                cached_data['detected_lang'],
                segments
            )
            
        # 비디오 임베딩
        if not self.worker.settings['srt_only'] and not self.worker.is_cancelled:
            self.embed_subtitles(
                video_path,
                cached_data['subtitles'],
                cached_data['detected_lang']
            )
            
        if not self.worker.is_cancelled:
            output_path = list(cached_data['subtitles'].values())[0]
            self.worker.safe_emit(self.worker.fileCompleted, video_name, output_path)
            self.worker.safe_emit(self.worker.progress, video_name, 100, "완료!")
            
    def process_translations(self, video_path, audio_paths, detected_lang, segments):
        """번역 처리"""
        if self.worker.is_cancelled:
            return
            
        video_name = filename(video_path)
        total_langs = len(self.worker.settings['languages'])

        # 번역 모듈 임포트 (지연 로딩)
        try:
            from auto_subtitle_llama.cli import translates
        except ImportError:
            self.worker.safe_emit(self.worker.log, "❌ 번역 모듈 로드 실패")
            return

        # mBART 소스 언어 코드 가져오기
        current_lang = LANG_CODE_MAPPER.get(detected_lang, [])
        source_mbart_code = current_lang[1] if len(current_lang) > 1 else "en_XX"
         
        for i, target_lang in enumerate(self.worker.settings['languages']):
            if self.worker.is_cancelled:
                break
                
            # 같은 언어는 건너뛰기
            if target_lang.startswith(detected_lang):
                continue
                
            # 캐시 확인
            cache_path = self.worker.get_cache_path(video_path, target_lang)
            if os.path.exists(cache_path):
                self.worker.safe_emit(
                    self.worker.progress,
                    video_name,
                    60 + (i * 30 // total_langs),
                    f"{target_lang} 캐시 로드 중..."
                )
                continue
                
            # 번역 수행
            progress = 60 + (i * 30 // total_langs)
            self.worker.safe_emit(self.worker.progress, video_name, progress, f"{target_lang} 번역 중...")
            
            try:
                if self.worker.is_cancelled:
                    break
                    
                self.worker.safe_emit(self.worker.log, f"{target_lang} 번역 시작...")

                # segments 깊은 복사
                translated_segments = copy.deepcopy(segments)
                
                # 텍스트 배치 추출 및 번역
                text_batch = get_text_batch(translated_segments)
                # 번역 실행 (싱글톤 모델 사용)
                translated_batch = translates(
                    translate_to=target_lang, 
                    text_batch=text_batch, 
                    source_lang=source_mbart_code
                )
                
                if self.worker.is_cancelled:
                    break
                    
                # 번역된 텍스트로 segments 업데이트
                translated_segments = replace_text_batch(translated_segments, translated_batch)
                
                # 번역된 SRT 저장
                translated_srt_path = os.path.join(
                    self.worker.settings['output_dir'], 
                    f"{video_name}_{target_lang.split('_')[0]}.srt"
                )
                
                with open(translated_srt_path, "w", encoding="utf-8") as srt_file:
                    write_srt(translated_segments, srt_file)
                
                translated_subtitles = {video_path: translated_srt_path}
                
                # 번역 결과 캐시
                with open(cache_path, 'wb') as f:
                    pickle.dump({
                        'subtitles': translated_subtitles,
                        'segments': translated_segments
                    }, f)
                    
                self.worker.safe_emit(self.worker.log, f"✓ {target_lang} 번역 완료")
                
                # 비디오 임베딩
                if not self.worker.settings['srt_only'] and not self.worker.is_cancelled:
                    self.embed_subtitles(
                        video_path, translated_subtitles, detected_lang, target_lang
                    )
                    
            except Exception as e:
                if not self.worker.is_cancelled:
                    self.worker.safe_emit(self.worker.log, f"❌ {target_lang} 번역 실패: {str(e)}")
                    
    def embed_subtitles(self, video_path, srt_paths, detected_lang, target_lang=None):
        """비디오에 자막 임베딩"""
        if self.worker.is_cancelled:
            return
            
        video_name = filename(video_path)
        self.worker.safe_emit(self.worker.progress, video_name, 90, "비디오에 자막 추가 중...")
        
        # 출력 파일명 생성
        lang_suffix = f"_{detected_lang}"
        if target_lang:
            lang_suffix += f"2{target_lang.split('_')[0]}"
            
        out_filename = f"{video_name}_subtitled{lang_suffix}.mp4"
        out_path = os.path.join(self.worker.settings['output_dir'], out_filename)
        
        try:
            if self.worker.is_cancelled:
                return
                
            srt_path = list(srt_paths.values())[0]
            
            video = ffmpeg.input(video_path)
            audio = video.audio
            
            # 자막 스타일 적용
            subtitle_style = (
                "FallbackName=NanumGothic,"
                "OutlineColour=&H40000000,"
                "BorderStyle=3,"
                "Fontsize=24,"
                "MarginV=20"
            )
            
            ffmpeg.concat(
                video.filter('subtitles', srt_path, 
                           force_style=subtitle_style, 
                           charenc="UTF-8"),
                audio,
                v=1,
                a=1
            ).output(out_path).run(quiet=True, overwrite_output=True)
            
            self.worker.safe_emit(self.worker.log, f"✓ 자막 임베딩 완료: {out_filename}")
            
        except Exception as e:
            if not self.worker.is_cancelled:
                self.worker.safe_emit(self.worker.log, f"❌ 자막 임베딩 실패: {str(e)}")
                
    def cleanup_temp_files(self):
        """임시 파일 정리"""
        temp_dir = tempfile.gettempdir()
        
        for video_path in self.worker.video_paths:
            audio_filename = f"{filename(video_path)}.wav"
            audio_path = os.path.join(temp_dir, audio_filename)
            if os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                    print(f"임시 파일 삭제: {audio_path}")
                except Exception as e:
                    print(f"임시 파일 삭제 실패: {e}")