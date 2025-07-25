import os
from typing import Iterator, TextIO, List
import threading

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