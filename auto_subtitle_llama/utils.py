import os
from typing import Iterator, TextIO, List
import threading
import difflib
import re

LANG_CODE_MAPPER = {
    "en": ["english", "en_XX"],
    "zh": ["chinese", "zh_CN"],
    "de": ["german", "de_DE"],
    "es": ["spanish", "es_XX"],
    "ru": ["russian", "ru_RU"],
    "ko": ["korean", "ko_KR"],
    "fr": ["french", "fr_XX"],
    "ja": ["japanese", "ja_XX"],
    "pt": ["portuguese", "pt_XX"],
    "tr": ["turkish", "tr_TR"],
    "pl": ["polish", "pl_PL"],
    "nl": ["dutch", "nl_XX"],
    "ar": ["arabic", "ar_AR"],
    "sv": ["swedish", "sv_SE"],
    "it": ["italian", "it_IT"],
    "id": ["indonesian", "id_ID"],
    "hi": ["hindi", "hi_IN"],
    "fi": ["finnish", "fi_FI"],
    "vi": ["vietnamese", "vi_VN"],
    "he": ["hebrew", "he_IL"],
    "uk": ["ukrainian", "uk_UA"],
    "cs": ["czech", "cs_CZ"],
    "ro": ["romanian", "ro_RO"],
    "ta": ["tamil", "ta_IN"],
    "no": ["norwegian", ""],
    "th": ["thai", "th_TH"],
    "ur": ["urdu", "ur_PK"],
    "hr": ["croatian", "hr_HR"],
    "lt": ["lithuanian", "lt_LT"],
    "ml": ["malayalam", "ml_IN"],
    "te": ["telugu", "te_IN"],
    "fa": ["persian", "fa_IR"],
    "lv": ["latvian", "lv_LV"],
    "bn": ["bengali", "bn_IN"],
    "az": ["azerbaijani", "az_AZ"],
    "et": ["estonian", "et_EE"],
    "mk": ["macedonian", "mk_MK"],
    "ne": ["nepali", "ne_NP"],
    "mn": ["mongolian", "mn_MN"],
    "kk": ["kazakh", "kk_KZ"],
    "sw": ["swahili", "sw_KE"],
    "gl": ["galician", "gl_ES"],
    "mr": ["marathi", "mr_IN"],
    "si": ["sinhala", "si_LK"],
    "km": ["khmer", "km_KH"],
    "af": ["afrikaans", "af_ZA"],
    "ka": ["georgian", "ka_GE"],
    "gu": ["gujarati", "gu_IN"],
    "lb": ["luxembourgish", "ps_AF"],
    "tl": ["tagalog", "tl_XX"],
}


def str2bool(string):
    string = string.lower()
    str2val = {"true": True, "false": False}

    if string in str2val:
        return str2val[string]
    else:
        raise ValueError(
            f"Expected one of {set(str2val.keys())}, got {string}")


def format_timestamp(seconds: float, always_include_hours: bool = False):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def write_srt(transcript: Iterator[dict], file: TextIO):
    for i, segment in enumerate(transcript, start=1):
        print(
            f"{i}\n"
            f"{format_timestamp(segment['start'], always_include_hours=True)} --> "
            f"{format_timestamp(segment['end'], always_include_hours=True)}\n"
            f"{segment['text'].strip().replace('-->', '->')}\n",
            file=file,
            flush=True,
        )


def filename(path):
    return os.path.splitext(os.path.basename(path))[0]

class TranslatorManager:
    """번역 모델 싱글톤 매니저"""
    _instance = None
    _lock = threading.Lock()
    _model = None
    _tokenizer = None
    
    def __new__(cls):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_translator(self):
        """번역 모델과 토크나이저 반환"""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._load_model()
        return self._model, self._tokenizer
    
    def _load_model(self):
        """모델 로드 (한 번만 실행)"""
        print("Loading translation model... (this may take a while)")
        from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
        self._model = MBartForConditionalGeneration.from_pretrained("SnypzZz/Llama2-13b-Language-translate")
        self._tokenizer = MBart50TokenizerFast.from_pretrained("SnypzZz/Llama2-13b-Language-translate", src_lang="en_XX")
        print("Translation model loaded successfully!")


def load_translator(source_lang="en_XX"):
    from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
    model = MBartForConditionalGeneration.from_pretrained("SnypzZz/Llama2-13b-Language-translate")
    tokenizer = MBart50TokenizerFast.from_pretrained("SnypzZz/Llama2-13b-Language-translate", src_lang=source_lang)
    return model, tokenizer

def get_text_batch(segments:List[dict]):
    text_batch = []
    for i, segment in enumerate(segments):
        text_batch.append(segment['text'])
    return text_batch

def replace_text_batch(segments:List[dict], translated_batch:List[str]):
    for i, segment in enumerate(segments):
        segment['text'] = translated_batch[i]
    return segments

def remove_duplicate_segments(segments: List[dict], similarity_threshold: float = 0.85) -> List[dict]:
    """중복된 자막 세그먼트 제거
    
    Args:
        segments: Whisper가 생성한 세그먼트 리스트
        similarity_threshold: 유사도 임계값 (0~1, 기본값 0.85)
    
    Returns:
        중복이 제거된 세그먼트 리스트
    """
    if not segments:
        return segments
    
    cleaned_segments = []
    last_text = ""
    consecutive_duplicates = 0
    
    for i, segment in enumerate(segments):
        current_text = segment['text'].strip()
        
        # 텍스트 유사도 계산
        similarity = difflib.SequenceMatcher(None, last_text, current_text).ratio()
        
        # 완전히 같거나 매우 유사한 경우
        if similarity >= similarity_threshold and current_text:
            consecutive_duplicates += 1
            # 3번 이상 연속으로 중복되면 건너뛰기
            if consecutive_duplicates >= 2:
                print(f"중복 제거: [{segment['start']:.3f} --> {segment['end']:.3f}] {current_text}")
                continue
        else:
            consecutive_duplicates = 0
        
        cleaned_segments.append(segment)
        last_text = current_text
    
    print(f"중복 제거 완료: {len(segments)}개 → {len(cleaned_segments)}개")
    
    return cleaned_segments

def advanced_remove_duplicates(segments: List[dict]) -> List[dict]:
    """고급 중복 제거 - 시간과 내용을 모두 고려
    
    1. 연속된 중복 제거
    2. 짧은 반복 패턴 제거
    3. 시간 간격을 고려한 병합
    """
    if not segments:
        return segments
    
    cleaned_segments = []
    skip_until = -1
    
    for i, segment in enumerate(segments):
        if i < skip_until:
            continue
            
        current_text = segment['text'].strip()
       
        # 너무 짧은 텍스트는 노이즈일 가능성이 높음
        if len(current_text) < 3 and not current_text.isalnum():
            continue
        
        # 앞으로 최대 10개 세그먼트 확인
        duplicate_indices = [i]
        for j in range(i + 1, min(i + 10, len(segments))):
            next_text = segments[j]['text'].strip()
            similarity = difflib.SequenceMatcher(None, current_text, next_text).ratio()
            
            # 유사도가 높고 시간이 연속적인 경우
            if similarity > 0.85:
                time_gap = segments[j]['start'] - segments[j-1]['end']
                if time_gap < 0.5:  # 0.5초 이내
                    duplicate_indices.append(j)
                else:
                    break
            else:
                break
        
        # 3개 이상 연속 중복이면
        if len(duplicate_indices) >= 3:
            # 첫 번째 세그먼트의 시작 시간과 마지막 세그먼트의 종료 시간으로 병합
            merged_segment = segment.copy()
            merged_segment['end'] = segments[duplicate_indices[-1]]['end']
            cleaned_segments.append(merged_segment)
            skip_until = duplicate_indices[-1] + 1
            print(f"연속 중복 {len(duplicate_indices)}개 병합: [{merged_segment['start']:.3f} --> {merged_segment['end']:.3f}] {current_text[:30]}...")
        else:
            cleaned_segments.append(segment)
   
    # 추가 정리: 너무 짧은 간격의 동일 텍스트 제거
    final_segments = []
    text_last_seen = {}
    
    for segment in cleaned_segments:
        text = segment['text'].strip()
        current_time = segment['start']
        
        # 이전에 본 텍스트인 경우
        if text in text_last_seen:
            last_time = text_last_seen[text]
            # 5초 이내에 같은 텍스트가 나왔다면 제거
            if current_time - last_time < 5.0:
                print(f"짧은 간격 중복 제거: [{segment['start']:.3f}] {text[:30]}...")
                continue
        
        text_last_seen[text] = current_time
        final_segments.append(segment)
    
    print(f"고급 중복 제거 완료: {len(segments)}개 → {len(final_segments)}개")
    return final_segments


# ============= 병렬 처리를 위한 새로운 함수들 =============

def merge_overlapping_segments(segments: List[dict], overlap_threshold: float = 0.1) -> List[dict]:
    """시간이 겹치는 세그먼트 병합"""
    if not segments:
        return segments
    
    merged = [segments[0]]
    
    for current in segments[1:]:
        last = merged[-1]
        
        # 시간이 겹치면 병합
        if current['start'] < last['end'] + overlap_threshold:
            # 텍스트 병합 (중복 제거)
            if current['text'].strip() != last['text'].strip():
                last['text'] = last['text'].rstrip() + ' ' + current['text'].lstrip()
            last['end'] = max(last['end'], current['end'])
        else:
            merged.append(current)
    
    return merged


def find_best_split_point(text: str, max_length: int = 80) -> int:
    """텍스트의 최적 분할 지점 찾기"""
    if len(text) <= max_length:
        return len(text)
    
    # 문장 끝 찾기 (. ? !)
    sentence_end = max(
        text.rfind('.', 0, max_length),
        text.rfind('?', 0, max_length),
        text.rfind('!', 0, max_length)
    )
    if sentence_end > 0:
        return sentence_end + 1
    
    # 쉼표 찾기
    comma = text.rfind(',', 0, max_length)
    if comma > 0:
        return comma + 1
    
    # 공백 찾기
    space = text.rfind(' ', 0, max_length)
    if space > 0:
        return space + 1
    
    return max_length


def adjust_timestamps(segments: List[dict], offset: float) -> List[dict]:
    """세그먼트의 타임스탬프를 오프셋만큼 조정"""
    adjusted = []
    for segment in segments:
        adjusted_segment = segment.copy()
        adjusted_segment['start'] += offset
        adjusted_segment['end'] += offset
        adjusted.append(adjusted_segment)
    return adjusted