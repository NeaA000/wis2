"""
비디오 처리 관련 로직 - 실시간 번역 통합 버전
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
    """비디오 처리 전담 클래스 - 실시간 번역 지원"""
    
    def __init__(self, worker):
        self.worker = worker
        self.model = None
        self.realtime_translations = {}  # 실시간 번역 결과 저장
        
    def detect_language(self, video_path):
        """비디오의 언어 감지 (캐시 확인 포함)"""
        # 캐시에서 확인
        cache_path = self.worker.get_cache_path(video_path)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    return cached_data.get('detected_lang')
            except:
                pass
        
        # 캐시가 없으면 None 반환 (나중에 감지)
        return None
        
    def save_realtime_translations(self, video_name, target_lang, segments):
        """실시간 번역 결과 저장"""
        key = f"{video_name}_{target_lang}"
        if key not in self.realtime_translations:
            self.realtime_translations[key] = []
        self.realtime_translations[key].extend(segments)
        
    def get_realtime_translations(self, video_name, target_lang):
        """실시간 번역 결과 가져오기"""
        key = f"{video_name}_{target_lang}"
        return self.realtime_translations.get(key, [])
        
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
        """단일 비디오 처리 - 실시간 번역 통합"""
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
            
            # 실시간 번역 결과 확인 및 통합
            if self.worker.realtime_translation_enabled and self.worker.progress_parser.streaming_translator:
                # 감지된 세그먼트 가져오기
                detected_segments = self.worker.progress_parser.get_detected_segments()
                
                # 실시간 번역 통계
                stats = self.worker.progress_parser.streaming_translator.get_stats()
                self.worker.safe_emit(
                    self.worker.log,
                    f"실시간 번역 완료 - 처리: {stats['translated_segments']}개, "
                    f"캐시 사용: {stats['cache_hits']}개"
                )
                
                # 번역 결과는 이미 realtimeTranslation 시그널로 전송됨
                # 여기서는 파일로 저장만 수행
            
        
            # 병렬 처리에서 번역이 이미 완료된 경우
            if isinstance(result, dict) and 'translations' in result:
                # 번역된 결과 저장
                for lang_code, trans_segments in result['translations'].items():
                    if trans_segments:
                    # 중복 제거 (병렬 처리 시 오버랩 구간 처리)
                        if self.worker.settings.get('advanced_duplicate_removal', True):
                            trans_segments = advanced_remove_duplicates(trans_segments)
                        else:
                            trans_segments = remove_duplicate_segments(trans_segments)
            
                        trans_srt_path = os.path.join(
                            self.worker.settings['output_dir'],
                            f"{video_name}_{lang_code.split('_')[0]}.srt"
                        )
                        with open(trans_srt_path, "w", encoding="utf-8") as srt_file:
                            write_srt(trans_segments, srt_file)
            
                        self.worker.safe_emit(self.worker.log, f"✓ {lang_code} 자막 저장: {trans_srt_path}")
            
                        # 비디오 임베딩
                        if not self.worker.settings['srt_only']:
                            self.embed_subtitles(
                                video_path, 
                                {video_path: trans_srt_path}, 
                                detected_lang, 
                                lang_code
                            )
                
                # result가 dict인 경우 segments 추출
                if isinstance(result, dict):
                    result = {'segments': result.get('segments', [])}
                
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
                
            # 6. 번역 처리 (실시간 번역이 없거나 비활성화된 경우)
            if (self.worker.settings['translate'] and 
                self.worker.settings['languages'] and
                not (isinstance(result, dict) and 'translations' in result) and
                not self.worker.realtime_translation_enabled):
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
        """음성 인식 수행 - 실시간 번역 지원"""
        # 언어 감지
        self.worker.safe_emit(self.worker.progress, video_name, 30, "언어 감지 중...")
        
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio, self.model.dims.n_mels).to(self.model.device)
        _, probs = self.model.detect_language(mel)
        detected_lang = max(probs, key=probs.get)
        self.worker.safe_emit(self.worker.log, f"감지된 언어: {detected_lang}")
        
        # 실시간 번역기가 아직 초기화되지 않았다면 여기서 초기화
        if (self.worker.realtime_translation_enabled and 
            self.worker.settings.get('translate') and 
            self.worker.settings.get('languages') and
            not self.worker.progress_parser.streaming_translator):
            
            current_lang = LANG_CODE_MAPPER.get(detected_lang, [])
            source_mbart_code = current_lang[1] if len(current_lang) > 1 else "en_XX"
            
            self.worker.progress_parser.init_streaming_translator(
                self.worker.settings['languages'],
                source_mbart_code
            )

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
        
        # 설정된 최소 시간 이상이고 병렬 처리 설정이 켜져있으면 병렬 처리
        min_duration = self.worker.settings.get('parallel_min_duration', 1800)
        if duration > min_duration and self.worker.settings.get('parallel_processing', True):
            self.worker.safe_emit(
                self.worker.log, 
                f"긴 오디오 감지 ({duration//60:.0f}분). 병렬 처리 시작..."
            )
            
            # 병렬 처리에서도 실시간 번역 지원
            return self._process_parallel_with_streaming(audio_path, video_name, detected_lang, duration)
            
        else:
            # 30분 미만 또는 병렬 처리 비활성화 시 단일 처리
            return self._transcribe_single(audio_path, video_name, detected_lang)
    
    def _transcribe_single(self, audio_path, video_name, detected_lang):
        """단일 스레드 음성 인식 - 실시간 번역 지원"""
        self.worker.safe_emit(self.worker.progress, video_name, 40, "음성 인식 시작...")
        
        # 진행률 파서의 플래그 설정
        self.worker.progress_parser.is_transcribing = True
        self.worker.progress_parser.transcribe_start_time = time.time()
        
        # 오디오 길이 설정
        try:
            probe = ffmpeg.probe(audio_path)
            duration = float(probe['streams'][0]['duration'])
            self.worker.progress_parser.set_audio_duration(duration)
        except:
            pass
        
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
            
    def _process_parallel_with_streaming(self, audio_path, video_name, detected_lang, duration):
        """병렬 처리 + 실시간 번역"""
        try:
            from gui.parallel_processor import process_long_video_parallel
        except ImportError:
            self.worker.safe_emit(self.worker.log, "❌ 병렬 처리 모듈 로드 실패, 단일 처리로 전환")
            return self._transcribe_single(audio_path, video_name, detected_lang)
        
        # 병렬 처리 시작 알림
        num_workers = self.worker.settings.get('num_workers', 3)
        self.worker.safe_emit(
            self.worker.log, 
            f"병렬 처리 시작: {num_workers}개 워커 사용"
        )
        
        # 워커 진행률 초기화
        self.worker.safe_emit(self.worker.initWorkerProgress, num_workers)
        
        # 진행률 콜백 함수
        def progress_callback(message, percent):
            if isinstance(message, dict) and message.get('type') == 'worker_progress':
                # 워커별 진행률 업데이트
                self.worker.safe_emit(
                    self.worker.workerProgress,
                    message['worker_id'],
                    message['chunk_index'],
                    message['progress'],
                    message.get('status', '처리 중')
                )
            elif isinstance(message, dict) and message.get('type') == 'log':
                # 로그 메시지 전달
                self.worker.safe_emit(self.worker.log, message['message'])
            else:
                # 전체 진행률
                self.worker.safe_emit(
                    self.worker.progress, 
                    video_name, 
                    40 + int(percent * 0.4),  # 40~80% 구간 사용
                    message
                )
        
        # video_path 가져오기
        video_path = self.worker.current_video_path
        
        # 번역 설정 추가
        task_args = {
            'task': 'transcribe',
            'language': detected_lang,
            'realtime_translation': self.worker.realtime_translation_enabled,  # 실시간 번역 플래그 추가
            'realtime_log': self.worker.settings.get('realtime_log', True)  # 실시간 로그 설정 추가
        }
        
        # 병렬 처리에서도 번역하도록 설정 (실시간이 아닌 경우)
        if self.worker.settings.get('translate') and self.worker.settings.get('languages'):
            # mBART 소스 언어 코드
            current_lang = LANG_CODE_MAPPER.get(detected_lang, [])
            source_mbart_code = current_lang[1] if len(current_lang) > 1 else "en_XX"
            
            task_args['translate_languages'] = self.worker.settings['languages']
            task_args['source_lang'] = source_mbart_code
        
        result = process_long_video_parallel(
            video_path=video_path,
            audio_path=audio_path,
            model_name=self.worker.settings['model'],
            task_args=task_args,
            num_workers=self.worker.settings.get('parallel_workers', 3),
            chunk_duration=self.worker.settings.get('chunk_duration', 1800),
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
            'language': detected_lang,
            'translations': result.get('translations', {})
        }
        
        self.worker.safe_emit(
            self.worker.log, 
            f"✓ 병렬 처리 완료: {len(result['segments'])}개 세그먼트"
        )
        
        return formatted_result, detected_lang
        
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

        # 병렬 번역기 사용
        from gui.worker.parallel_translator import ParallelTranslator
        translator = ParallelTranslator(self.worker)

        # mBART 소스 언어 코드 가져오기
        current_lang = LANG_CODE_MAPPER.get(detected_lang, [])
        source_mbart_code = current_lang[1] if len(current_lang) > 1 else "en_XX"
         
        # 캐시되지 않은 언어만 필터링
        languages_to_translate = []
        for lang in self.worker.settings['languages']:
            # 언어 비교 버그 수정: mBART 형식으로 비교
            if lang == source_mbart_code:
                self.worker.safe_emit(self.worker.log, f"✓ {lang}는 원본과 같은 언어이므로 건너뜁니다")
                continue
            cache_path = self.worker.get_cache_path(video_path, lang)
            if not os.path.exists(cache_path):
                languages_to_translate.append(lang)
        
        if not languages_to_translate:
            self.worker.safe_emit(self.worker.log, "모든 번역이 캐시에 있음")
            return
        
        # 병렬 번역 실행
        self.worker.safe_emit(self.worker.progress, video_name, 60, f"{len(languages_to_translate)}개 언어 병렬 번역 시작...")
        
        translation_results = translator.translate_multiple_languages(
            segments,
            languages_to_translate,
            source_mbart_code,
            video_name
        )
        
        # 결과 저장 및 임베딩
        for target_lang, translated_segments in translation_results.items():
            if translated_segments and not self.worker.is_cancelled:
                # SRT 저장
                translated_srt_path = os.path.join(
                    self.worker.settings['output_dir'], 
                    f"{video_name}_{target_lang.split('_')[0]}.srt"
                )
                
                with open(translated_srt_path, "w", encoding="utf-8") as srt_file:
                    write_srt(translated_segments, srt_file)
                
                # 캐시 저장
                cache_path = self.worker.get_cache_path(video_path, target_lang)
                with open(cache_path, 'wb') as f:
                    pickle.dump({
                        'subtitles': {video_path: translated_srt_path},
                        'segments': translated_segments
                    }, f)
                
                # 비디오 임베딩
                if not self.worker.settings['srt_only']:
                    self.embed_subtitles(
                        video_path, 
                        {video_path: translated_srt_path}, 
                        detected_lang, 
                        target_lang
                    )

        # 번역 완료 후 메모리 정리
        if 'translator' in locals():
            try:
                from auto_subtitle_llama.utils import TranslatorManager
                manager = TranslatorManager()
                if hasattr(manager, '_model') and manager._model is not None:
                    import gc
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
            except:
                pass
                    
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